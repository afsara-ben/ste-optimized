# Batched Expert-in-the-Loop Replay Training — Portable Plan

**Created:** 2026-07-19 (UTC)
**Updated:** 2026-07-20 05:58 UTC — **TWO 50-UPDATE RUNS COMPLETED WITHIN THE
ONE-HOUR ENVELOPE; SEED-42 UPDATE 45 AT ALPHA 0.73 PASSED THE FULL 40-ROW
VALIDATION PANEL.** The
plan below is now code: a
self-contained package at `ste-optimized/` (repo root; own `pyproject.toml`, package
`ste_optimized`, no dependency on the `emotionsteer` package — designed to be moved
into its own git repo). Status: the bounded `smoke` workflow is implemented; 63 CPU
tests pass; three real-WAV GPU parity components pass; differentiable
Whisper-large-v3-turbo token NLL has an isolated nonzero waveform-gradient smoke; and
one real angry-minus-neutral contrast completed the full FlashAttention Qwen → replay
→ reference-conditioned codec → emotion2vec/WavLM/Whisper → backward path and exported
a learned transform and steering payload. A measured `K=16, M=2, B=32,
chunk_rows=4` update established the configuration used by the bounded one-hour run.
Two seeds then completed 50 updates in parallel in 24m55s/25m09s. Exhaustive
checkpoint×alpha screening followed by a 40-contrast, 276-word panel selected seed 42,
update 45, alpha 0.73: all aggregate emotion/speaker/ISR/WER gates passed. It improved
angry probability on 29/40 matched rows (72.5%), so this is evidence of held-out
generalization, **not** proof that every angry contrast pair improves. §8 records the
exact evidence, artifacts, and limitations.
Section 6 gives the commands and maps each plan phase to its module. §3.1 contains the
exact loss formula, two-pass T(v) recomputation, ASR path, and gradient-path semantics
including what STE captures versus drops.
**2026-07-20:** five implementation bugs fixed after external review (per-chunk
T(v) recompute; reference-conditioned decode+trim mirroring native inference;
codec dtype cast; explicit codec freeze + full gradient-contract tripwire;
subtalker warpers in the residual STE derivative) plus an optional
`steer_frame0_predictor` flag (default off). **Initial milestone simplified:**
one angry transform, one fixed batch, and one acceptance test — the complete
reference-conditioned waveform loss must produce a finite nonzero expert-only
gradient in T_θ, change its parameters, decrease under repeated optimization,
and leave every frozen model gradient-free — then multi-chunk and native/replay
parity, BEFORE any DDP, bucketing, or other emotions (see `ste-optimized/README.md`).
The finite-gradient, parameter-change, frozen-model, and multi-chunk execution pieces
have passed. Across the completed run, the ten-update mean emotion loss decreased
substantially, while speaker similarity and Whisper NLL worsened late; validation
therefore selected an earlier checkpoint rather than the final one. The combined
weighted scalar was not logged separately and should be added before claiming a
monotone total-loss curve.
**Derived from:** `BATCHED_REPLAY_PLAN_july19.md` (machine-specific measurements and
file-level work items live there; this document is the machine-independent, executable
version). Objective and estimator are those of the original replay program; only the
batching design and training hygiene change.

---

## 1. Goal

Train a rank-16 low-rank transform `T_θ(v) = v + U(Dv)` (identity init: orthonormal
random `D`, zero `U`; 32,768 params) mapping a layer-15 contrast `v ∈ R^1024` to an
improved steering vector `u`, by **generating steered audio during training and
backpropagating frozen expert losses** (emotion2vec + WavLM +
Whisper-large-v3-turbo token NLL) through a fixed-hard-STE replay graph into `T_θ`
only. The ASR term preserves the requested transcript. Qwen, codec, and all three
experts stay frozen; no RL; no reward models.

## 2. Prerequisites (any machine)

- **GPU:** ≥ 24 GB VRAM (Qwen-0.6B + both experts resident together; pass-2 chunks are
  checkpointed and a few GB). More VRAM ⇒ wider generation batches, nothing else.
  A second GPU is used for a second seed, never for intra-update DDP (measured 1.13×).
- **Software:** Python ≥ 3.12, torch ≥ 2.7 + CUDA, `qwen-tts == 0.1.1`,
  transformers ≥ 4.57, flash-attn (fallback sdpa — record which), funasr + modelscope,
  soundfile, librosa, jiwer. Record all versions in artifacts.
- **Models (pinned):** `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
  @ `5d83992436eae1d760afd27aff78a71d676296fc` (bundles the 12 Hz speech tokenizer);
  `emotion2vec/emotion2vec_plus_large` and `microsoft/wavlm-base-plus-sv`
  (**inside the training loss**, frozen); `openai/whisper-large-v3-turbo`
  @ `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` (**inside the training loss** as
  differentiable teacher-forced token NLL, and used for decoded validation WER;
  frozen in both roles).
- **Data per emotion:** ~200 train pair contrasts + neutral-base catalog, ≥18
  (target 40) validation contrasts, extracted with the frozen extraction pipeline;
  per-pair extraction-time expert scores retained as fixed loss weights. Canonical
  speaker partition (mandatory): train 0011/0014/0017/0020, val 0012/0016,
  test 0013/0019, reserve 0015/0018.

## 3. The batched update (Algorithm)

One optimizer update = `K` contrasts × `M` bases = `R` rows. The measured one-hour
RTX A6000 plan uses **`K=16, M=2, R=B=32, chunk_rows=4`**: B=32 is the observed
generation-throughput knee, while K=16 favors contrast diversity and M=2 retains two
independent cross-speaker neutral bases per contrast (§4 and §8).

```text
INPUT: contrast batch {v_1..v_K} (shuffled without replacement, per-speaker balanced),
       for each contrast M cross-speaker neutral bases (base speaker ≠ contrast
       speaker; resampled every epoch), current transform T_θ

PASS 1 — hard sampling, no gradients, ONE batched generate over all R rows:
  1  u_i ← T_θ(v_i).detach();  every row of contrast i carries u_i
  2  steering: per-row vector applied at layer 15 on DECODE steps only
     (prefill unsteered), each steered state renormalized to its original L2
     norm; optional `steer_frame0_predictor` flag additionally steers ONLY the
     last prompt position (the frame-0 predictor), symmetrically here, in
     replay, and in evaluation — default OFF (historical convention)
  3  prompts served from a base_id-keyed cache (encode once, reuse all epochs)
  4  frame cap ≈ 2 × p95 of natural code lengths (NOT an arbitrary large cap);
     bucket rows by expected length so one long row cannot stall short ones
  5  per-row EOS: a capped row FAILS ALONE and is dropped; accept the update if
     ≥ 80% of rows survive (renormalize weights, log failures); never retry
     inside the update
  OUTPUT: hard codes Y_r ∈ ℕ^{T_r×16} per surviving row

PASS 2 — gradients, in chunks of 4 rows with gradient accumulation for the
         measured B32 one-hour configuration:
  6  batched padded teacher-forced replay forward over the chunk's stored codes,
     with the SAME per-row steering and renormalization as pass 1, reproducing
     pass-1's logit processing (temperature, top-k/p, EOS suppression,
     repetition penalty from the stored prefix) so the STE derivative matches
     the distribution the codes were sampled from
  7  fixed-hard STE: y = one_hot(code) − sg(p) + p  (forward = sampled code,
     gradient = replay probability)
  8  REFERENCE-CONDITIONED differentiable codec decode: prepend the base's
     reference codes as constant one-hots, decode cat([ref; ŷ]) chunk-batched
     with activation checkpointing, trim the reference span proportionally —
     mirroring native inference (qwen3_tts_model.py:614-629) so the experts
     score the SAME waveform normal Qwen output produces
     → waveforms → batched emotion2vec loss + WavLM speaker loss + frozen
     Whisper-large-v3-turbo teacher-forced token NLL
  9  chunk loss = Σ rows {w_row · [emotion_loss + speaker_weight · speaker_loss]
                         + 0.2 · whisper_token_NLL};
     backward per chunk (accumulate). Whisper NLL is deliberately not multiplied
     by the contrast-confidence weight: every row has a known requested transcript.

STEP: clip grad-norm 1.0; AdamW step (lr 1e-3, wd 1e-4, 5% warmup + cosine);
      log per-phase wall time, loss components, grad/param norms, survival rate
```

### 3.1 Loss function and gradient path (exact)

```text
u_i        = T_θ(v_i)                     # pass 1 uses u_i.detach();
                                          # pass 2 RECOMPUTES T_θ(v_i) with grad —
                                          # same values, different autograd status
Y_{i,b}    ~ Generate(prompt_b, steer(u_i.detach()))          # pass 1, no gradients
p_t        = softmax(process(replay_logits_t(u_i)))           # pass 2; t indexes frames
ŷ_t        = onehot(Y_t) − sg(p_t) + p_t                      # fixed-hard STE
wav_{i,b}  = trim( CodecDecode_soft([onehot(ref_b); ŷ]) )     # frozen, differentiable
```

`process(·)` covers codebook 0's full pass-1 chain (repetition penalty,
suppress, min-new-tokens, temperature/top-k/top-p) AND, for the residual
codebooks 1-15, the subtalker's own warpers (temperature 0.9 / top-k 50 by
default) — the STE derivative must replay the distribution each codebook was
actually sampled from. The decode is reference-conditioned: the base's
reference codes `ref_b` enter as CONSTANT one-hots (no gradient) and the
reference span is trimmed proportionally, exactly as native inference decodes.
Inside a chunked update, `u_i = T_θ(v_i)` is recomputed FRESH per chunk —
each chunk's backward frees the graph it traverses, so a transform forward
shared across chunks would crash on the second chunk.

Notation note — the subscript t indexes frames MATHEMATICALLY; computationally pass 2
is BATCHED, not a per-frame loop: one padded teacher-forced talker forward per chunk
of rows produces the logits of every frame of every row simultaneously, one
`forward_sub_talker_finetune` call scores all rows' frames flattened, the codec
decodes the chunk's one-hot tensors together, and the experts score the chunk's
waveforms as one padded batch. The only frame-indexed construction is the
repetition-penalty "seen" mask — a constant of the stored trajectory, built under
no_grad; the penalty itself is then applied in one vectorised tensor op.

```text
L = (1/N) · Σ_{i,b ∈ surviving rows}
        { w_i · [ −log P_e2v(c | wav_{i,b})                   # emotion loss
                  + λ_spk · (1 − cos(xvec(wav_{i,b}), xvec(ref_b))) ] # speaker loss
          + λ_asr · NLL_whisper(text_b | wav_{i,b}) }         # ASR token loss
    + λ_id·L_identity + λ_cos·L_cosine + λ_norm·L_norm        # keep T(v) near v
```

The regularizers, averaged over the update's K contrast vectors (fp32; their
gradient reaches θ directly, no model in the path), with d = hidden size (1024):

```text
L_identity = mean_i  ‖T_θ(v_i) − v_i‖² / d          # small residual
L_cosine   = mean_i  ( 1 − cos(T_θ(v_i), v_i) )     # keep direction
L_norm     = mean_i  ( log( ‖T_θ(v_i)‖ / ‖v_i‖ ) )² # keep norm ratio near 1
```

Defaults: λ_spk = 1.0; **λ_asr = 0.2**; λ_id = λ_cos = λ_norm = 0.01; N = total
rows of the update
(across ranks if sharded); w_i = the pair's extraction-time target-emotion
probability — a CONSTANT, no expert gradients flow through it.
P_e2v = frozen emotion2vec head; xvec = frozen WavLM-SV embedding; ref_b = the base's
ICL reference recording. `NLL_whisper` is the per-row mean over requested transcript
tokens plus EOS; fixed language/task/timestamp control tokens are excluded. Its
torch-native log-mel frontend is essential: the standard NumPy-returning feature
extractor would detach the waveform and destroy this gradient path.

Gradient path (only θ trains): L → experts → waveform → codec decoder → ŷ
(∂ŷ/∂logits = ∂p/∂logits) → replay logits → talker layers above 15 → the
renormalised steering injection at layer 15 → u_i → θ, plus the direct regularizer
path. More explicitly, the ASR branch is `token NLL → frozen Whisper decoder →
frozen Whisper encoder → differentiable torch log-mel/STFT frontend → waveform`,
after which it joins the same codec/STE/replay path. Whisper parameters are frozen and
must remain gradient-free, but autograd through its operations to the waveform is
required. Decoded transcription and WER are non-differentiable evaluation only; WER
does not enter the training gradient. Every module on the training path is frozen but
differentiable except `T_θ`, the sole trainable module.

What the gradient captures vs drops: it knows how u shifts each frame's token
DISTRIBUTION on the frozen pass-1 trajectory (and how that changes the decoded
waveform and expert scores). It does NOT know how u would have changed WHICH tokens
got sampled — the sampled-token → next-step-input feedback edge is severed because
pass-2 inputs are frozen constants, and the EOS/length decision is likewise frozen
(duration ceiling). That feedback closes ACROSS updates instead: every pass 1
re-samples with the current u. Magnitude context: the severed paths carried most of
the exact gradient's norm (historically 392.6 exact vs 8.5 replay); the program's
gated bet is that the surviving term's direction is useful.

Seeds: everything deterministic per (seed, update, row identity); two training seeds
required for any claim.

## 4. Dataset size and base allocation (assessed 2026-07-19)

- **200 train contrasts/emotion is not overkill:** the transform edits a 16-dim
  subspace; 200 samples across 4 speakers is a modest ratio, and with batching an
  epoch costs minutes — shrinking the pool saves nothing. Keep the frozen-manifest /
  leakage-check machinery unchanged (cheap, standard).
- **Validation is the thin side:** gates were designed for 40 pairs; run the pilot on
  what exists but extend toward 40 before any freeze decision (bootstrap CIs at n≈18
  are wide).
- **Bases: 5 per contrast was an inheritance, not a requirement.** It dates from
  1-contrast-per-update designs where bases were the only averaging axis. In a batched
  update the gradient averages over all R rows regardless of grouping, so at fixed R
  the real trade is contrast diversity (the map's actual input data) vs repeated noisy
  evaluations of one input. Prefer diversity. The implemented, measured A6000
  one-hour schedule therefore uses **K=16 × M=2 = B32**, with `M=2` the floor
  (per-contrast diagnostics get too noisy below) and `chunk_rows=4` for the
  differentiable replay/codec/expert pass. `K=10 × M=4` remains a useful
  higher-replication calibration arm, and `M=5` only a historical-comparability arm.
  Bases are resampled each epoch, so coverage of each contrast's ~15 eligible
  cross-speaker bases accumulates within a few epochs anyway.

## 5. Calibration on a new machine (run FIRST; ~15 min)

```bash
ste-optimized calibrate -c configs/angry.yaml     # benchmarks 1-4 -> JSON report
ste-optimized train -c configs/angry.yaml --max-updates 10   # benchmark 5 (timed smoke)
```

Five micro-benchmarks fully determine the schedule. Reference values (RTX A6000-48GB,
bf16, flash-attn 2) in brackets — do not reuse them unmeasured:

1. Single-stream generation ms/frame [124 ms/frame].
2. Batched generation at B ∈ {8, 16, 32, 64}: frames/s and per-step wall
   [41 / 87 / 200 frames/s at 8/16/32; per-step flat ~132 ms]. Pick the knee.
3. Replay/TF fwd+bwd microbatch through the talker [~0.5 s at 16 rows × 192 pos].
4. Batched codec decode + emotion2vec + WavLM + Whisper forward/backward per row.
   The historical `[25 / 8 / 3 ms/row at B=8]` measurement excluded Whisper and is
   no longer a valid estimate of the implemented pass; use benchmark 5/full-update
   timing instead.
5. **10-update timed smoke** of the full loop — the only trustworthy source for the
   pass-2 backward (differentiable codec + experts, checkpointed) [single-row
   reference ~0.36 s at 50 frames; batched value must be measured, ±2× a priori].

Schedule formulas:

```text
t_update ≈ R × mean_frames / gen_frames_per_s   (pass 1)
         + n_chunks × t_chunk(measured in #5)   (pass 2)
         + ~1 s                                 (cached prompts, optimizer, logs)
epoch    = ceil(N_train / K) updates
one-hour pilot ≤ 50 updates; production follow-up ≤ 300 with early stop
```

Measured initial outcome (A6000, B32): one `K=16, M=2, chunk_rows=4` update took
`1.822s` prompt preparation + `17.021s` pass-1 generation + `13.409s` differentiable
pass 2 + `0.027s` optimizer = **32.279s** in the recorded phase timers (§8). At that
single-update rate, 50 training updates alone project to ~26.9 minutes, but sampling
length variance, checkpointing, cadence panels, startup, and final evaluation still
require a measured end-to-end run. That run is now complete (§8): full B32 updates
averaged `28.849s` (seed 42) and `29.212s` (seed 43), and the complete 50-update
training commands—including model load and five cadence panels—finished in `24m55s`
and `25m09s`. The original practical estimate of ~28–32 minutes was therefore
conservative. `configs/angry-1h.yaml` bounds training at 50 updates, 75 attempts, and
3000 seconds. Evaluation generation is deterministically chunked to at most B32.

The recorded run exposed four inefficient B2 epoch tails because 161 qualified
contrasts are not divisible by K=16. The sampler now carries an incomplete tail into
the next shuffled epoch, persists the adjusted order for exact resume, and prevents a
duplicate contrast inside a boundary-spanning batch. Future runs therefore retain
full B32 updates without dropping contrast coverage. With exhaustive two-seed
checkpoint×alpha screening and full validation, the measured end-to-end artifact
pipeline still completed in `59m52.7s` from the earliest training-directory creation
to the final selected-alpha report.

## 6. Workflow (implemented — commands are real)

Implementation map (package `ste_optimized`, folder `ste-optimized/`):

| Plan element | Module |
|---|---|
| Pass-1 batched generation, prompt cache by base_id, per-row EOS | `backend.py` |
| Per-row steering + renorm (decode-step and masked-replay), extraction capture | `hooks.py` |
| Fixed-hard STE + pass-1 logit-chain replay (rep-penalty 1.05, suppress, min-new-tokens, temp/top-k/top-p) | `ste.py` |
| Pass-2 padded teacher-forced replay, LEFT-padding (matches native generate), target gathering, subtalker scoring | `replay.py` |
| Differentiable soft-codec decode (chunked + checkpointed) | `codec.py` |
| Frozen emotion2vec + WavLM, batched, differentiable | `experts.py` |
| Frozen Whisper-large-v3-turbo token NLL with torch-native differentiable log-mel frontend | `whisper_asr.py` |
| Whisper-normalized decoded corpus WER with deterministic S/D/I counts | `wer_metrics.py` |
| ESD ingest, pair/base records, fail-closed leakage checks | `data.py` |
| Batched resumable mean-decode contrast extraction + fixed weights (nothing pre-extracted assumed) | `extraction.py` |
| K×M sampler, per-epoch base rotation, full boundary-spanning batches, exact resume | `sampling.py` |
| Update loop: survival policy, chunked accumulation, AdamW+warmup/cosine, phase timings, checkpoints, early stop | `training.py` |
| Cadence + memory-bounded full/checkpoint panels, cached/matched controls, bootstrap CI, checkpoint×alpha selection | `evaluation.py` |
| Micro-benchmarks (§5) | `calibrate.py` |
| Seed-parallel policy + optional torchrun `ddp_rows` row-sharding | `distributed.py` |

- **Stage 0 — initial real-contrast gradient/parameter and ASR-gradient acceptance
  COMPLETE; bounded angry validation acceptance COMPLETE.** The `smoke`
  command uses one extracted angry-minus-neutral contrast,
  broadcasts the same `T(v)` across four neutral training rows, replays in two-row
  chunks, sends decoded audio through emotion2vec, WavLM, and frozen Whisper token NLL
  (weight 0.2), rejects zero/non-finite transform gradients, and evaluates the learned
  vector on four disjoint neutral utterances. It is attempt/wall bounded and exports
  only after its held-out emotion/speaker/EOS/decoded-WER gates pass. The 2026-07-20
  run passed on an RTX A6000 with FlashAttention 2 (§8).
  Prompt assembly, bare/reference soft-codec, and emotion2vec parity also passed on
  a real WAV, and the Whisper branch independently produced a finite nonzero waveform
  gradient while its parameters stayed frozen. Two training seeds, exhaustive
  validation checkpoint/alpha selection, and the full angry panel are now recorded in
  §8. DDP (`ddp_rows`), length bucketing, other emotions, an independent emotion
  judge, true ASR-based ISR, multiple evaluation sampling seeds, and the untouched
  test split remain deferred. Native processed-logit parity for the replay
  approximation is also not yet a dedicated GPU assertion.
- **Stage 1 — data:**
  `ste-optimized build-data -c configs/angry.yaml --source /path/to/ESD` then
  `ste-optimized extract -c configs/angry.yaml --split train` (and `validation`);
  set `data.contrasts_path`. Extraction is batched (~25× over scalar) and resumable.
- **Stage 2 — gates:** the milestone gates above (acceptance → multi-chunk →
  parity, in order) + `calibrate` + timed full updates (§5). The B32 update and two
  sustained 50-update runs have measured the complete
  Qwen/replay/codec/three-expert/backward cost and cadence overhead.
- **Stage 3 — one-hour bounded train:** run up to 50 updates using the measured
  `K=16, M=2, B=32, chunk_rows=4` schedule, checkpoint every 5 updates, evaluate every
  10, and stop after 3000 training seconds or 75 attempts. Run two seeds, one GPU each:
  `CUDA_VISIBLE_DEVICES=0 ste-optimized train -c configs/angry-1h.yaml --seed 42 --output runs/angry-1h-s42 &`
  `CUDA_VISIBLE_DEVICES=1 ste-optimized train -c configs/angry-1h.yaml --seed 43 --output runs/angry-1h-s43 &`
  (optional row-sharding instead: `torchrun --standalone --nproc-per-node=2 -m
  ste_optimized train -c … --distributed ddp_rows`; historically ~1.13× — seed-parallel
  is the recommended use of extra GPUs.) Both commands completed (§8). If this bounded
  pilot is extended, retain the original ≤300-update early-stopped schedule rather
  than treating one hour as a convergence guarantee.
- **Stage 4 — validation gate:**
  `ste-optimized evaluate-checkpoints -c configs/angry-1h.yaml --run-dir
  runs/angry-1h-s42 --rows 12`, followed by the selected checkpoint on the full panel
  with `--rows` omitted. Limited panels are explicitly marked provisional, interleave
  contrast speakers, and use the same canonical contrast→base mapping as the full
  panel. Generation/expert scoring is chunked at B32; one Qwen/emotion/WavLM/Whisper
  load is reused across all snapshots; raw controls are reused per alpha. Gates:
  emotion-prob delta > 0 with 95%
  paired-bootstrap CI excluding 0; WavLM ≥ 0.85 and degradation ≤ 0.05; ISR ≥ 80% and
  ≥ control − 5 pp. Decoded corpus WER is compared to the **raw-vector arm at the
  same alpha**: only `WER_learned − WER_raw > +0.06` absolute fails, so a degradation
  of exactly `+0.06` passes. This is intentionally a one-sided preservation gate, not
  a claim that WER improved. Require at least 100 normalized reference words for the
  production gate. Checkpoint chosen on validation only; the test split will be
  touched once, after freezing a future final protocol.
- **Stage 5 — fallback:** teacher-forced likelihood-preference trainer
  (`TEACHER_FORCED_PREFERENCE_PLAN_PORTABLE.md`) if replay fails gates.
- **Stage 6 — replicate** per emotion; shared conditioned transform only after all pass.

## 7. Verification checklist (implemented in `ste-optimized/tests/`)

- **CPU unit tests (63, green):** `test_transform.py` (identity init, T(0)=0,
  saddle-avoidance, save/load, regularizers), `test_ste.py` (STE forward-hard /
  gradient-soft, history-dependent repetition penalty, min-new-tokens EOS
  suppression, top-k/temperature, gradient flow through the chain),
  `test_hooks.py` (prefill skipped, per-position renorm, masked positions only,
  capture respects per-row lengths, `steer_frame0_predictor` flag),
  `test_training_chunks.py` (freed-graph regression: the old shared-T(v)
  pattern raises; per-chunk recompute grads == single-backward grads),
  `test_sampling_data.py` (cross-speaker constraint, epoch coverage, base
  rotation, resume parity, quality filter, leakage fail-closed), plus bounded-loop,
  inference-tensor codec-prefix, smoke-gate, nonzero-gradient, and artifact-path
  regressions in `test_training_bounds.py`, `test_codec.py`, and `test_smoke.py`;
  `test_whisper_asr.py` covers the differentiable frontend, token masking/NLL, frozen
  model contract, and waveform gradient; `test_wer_metrics.py` covers normalization,
  deterministic S/D/I counts, corpus aggregation, empty hypotheses, minimum-word
  enforcement, and the inclusive `+0.06` matched-raw boundary;
  `test_checkpoint_evaluation.py` covers single-load snapshot evaluation, transform
  deduplication, alpha selection, matched controls, B32 evaluation chunking,
  provisional/full WER floors, and speaker-balanced limited panels. Sampler tests now
  also cover full boundary-spanning batches and exact resume after a carried tail;
  the cadence-loop regression rejects the `-1.0` failure sentinel as a best model.
- **GPU milestone gates (5, `pytest -m gpu`, in order):** the one-fixed-batch
  acceptance test (complete reference-conditioned loss; four assertions),
  multi-chunk accumulation, prompt-assembly parity, soft-codec parity (bare
  and reference-conditioned), emotion2vec-head parity. The isolated real-model
  Whisper waveform-gradient smoke and the end-to-end B32 update are recorded in §8.
- **End-to-end (after the milestone):** export → panel → gates vs raw /
  global / purifier / cached-activation baselines (`evaluate` CLI).

Defects caught by these tests so far: an autograd in-place violation in the
penalty-chain replay and a negative suppress-range index (during initial
implementation), plus five review-found bugs on 2026-07-20 (shared-graph
double-backward, missing reference-conditioned decode, fp32→bf16 codec crash,
unfrozen codec, unwarped residual STE derivative), and a PyTorch inference-tensor
in-place clamp in the real codec prefix — evidence the gates bite, not decoration.

## 8. Timestamped execution results

### 2026-07-20 01:28:01 UTC — review evidence (this workspace)

**Scope and changes made:** this review was read-only with respect to the training
implementation. No smoke-test, training-loop, evaluation, or model code was changed,
and no angry transform was trained. Documentation changes only: the stale CPU-test
count and Stage-0 status were corrected, and this timestamped record was added. The
proposed real-contrast smoke-test changes remain future work.

**Environment:** torch `2.7.1+cu126`; `torch.cuda.is_available() == False`;
CUDA device count `0`. Therefore the full Qwen → replay → codec → experts → backward
GPU milestone could not execute in this workspace.

**Commands and observed results:**

```text
PYTHONPATH=src pytest -q
27 passed, 5 deselected in 1.18s

PYTHONPATH=src pytest -q -m gpu tests/test_parity.py -rs
5 skipped in 1.14s — every test skipped because no CUDA device was available
```

**Observation:** the successful command above is a CPU unit-test run, not the angry
training smoke. It supports the CPU-level transform/hook/sampling/chunking plumbing
only. It does **not** show that an angry contrast was learned, that `T(v)` improved
anger on neutral utterances, or that expert gradients traversed the installed Qwen
and codec graph on GPU. No audio, trained checkpoint, or learned steering vector was
produced.

The currently written GPU acceptance test would still be insufficient for that
claim even on a CUDA machine: it supplies independent random Gaussian vectors rather
than one extracted angry-minus-neutral contrast shared across multiple neutral bases,
and its tail criterion can pass when only the combined emotion-plus-speaker loss (or
sampling noise) improves. The first real smoke must instead use one real contrast,
broadcast the same `T(v)` over a small batched neutral panel, run the complete Qwen
generation/replay/codec/expert/backward path, and separately require held-out angry
probability improvement subject to speaker-similarity and termination constraints.
It also needs an attempt or wall-time cap so a short smoke cannot hang or repeatedly
evaluate update zero after generation-survival failures.

**Current milestone status:** CPU gate passed; GPU plumbing gate unexecuted; real
angry-transform learning gate unimplemented and unexecuted. Stage 0 must not be called
complete until those last two gates pass and save their checkpoint, source contrast,
`T(v)`, metrics, and sample audio.

### 2026-07-20 02:27:56–02:28:50 UTC — real angry-transform smoke (PASS)

This entry supersedes the 01:28 UTC environment/status conclusion above. CUDA was
installed; the managed command sandbox had hidden `/dev/nvidia0`, `/dev/nvidia1`, and
`/dev/nvidiactl`. Running the GPU commands with device access exposed two NVIDIA RTX
A6000 GPUs. The verified runtime was Python 3.12.4, torch `2.7.1+cu126`, torch CUDA
12.6, NVIDIA driver `595.71.05`, and FlashAttention `2.8.3`. The initially cached
FlashAttention wheel had the wrong torch ABI; it was replaced with the official
`cu12torch2.7cxx11abiTRUE-cp312-linux_x86_64` wheel. The smoke fails closed unless
Qwen's model, talker, and code predictor all report `flash_attention_2`; all three did.

Changes implemented for the bounded proof:

- added `ste-optimized smoke`, `configs/angry-smoke.yaml`, and the local cached-ESD
  materializer;
- selected the real matched pair `0011:000254` (`"She laughed."`), extracted its
  layer-15 angry-minus-neutral vector, and stored the source pair and scores;
- fixed `K=1`, `M=4`: one native four-row Qwen generation per optimizer update, the
  same `T(v)` on every neutral row, two-row replay chunks, reference-conditioned soft
  codec decode, emotion2vec and WavLM losses, then backward into `T` only;
- added attempt/wall bounds, skipped-update accounting, checkpoint/final export,
  pre-clip gradient/parameter/steering deltas, full frozen-module tripwires, and an
  explicit rejection of a zero transform gradient;
- fixed the real-run PyTorch error caused by an in-place clamp on Qwen reference codes
  created under inference mode; and
- added held-out unsteered/raw controls, row-level directionality, speaker/EOS gates,
  alpha selection, WAV export, a learned-transform artifact, and a directly reusable
  1024-D steering payload. The current smoke code uses the requested lenient relative
  speaker bound (degradation ≤ 0.05); the successful run below passed the earlier,
  stricter ≤ 0.02 bound.

Commands and verification:

```text
HF_HUB_OFFLINE=1 STE_OPT_ATTN_IMPLEMENTATION=flash_attention_2 \
  STE_OPT_REF_WAV=data/esd-smoke-source/wav/0012_000332.wav \
  STE_OPT_REF_TEXT='where are you going?' \
  pytest -q -m gpu tests/test_parity.py \
  -k 'prompt_assembly or soft_codec or emotion2vec' -rs
3 passed, 2 deselected in 10.89s

pytest -q -m 'not gpu'
37 passed, 5 deselected in 2.13s

HF_HUB_OFFLINE=1 python -m ste_optimized smoke \
  -c configs/angry-smoke.yaml --pair-id 0011:000254 \
  --output runs/angry-smoke-20260720T022733Z
status: pass; success gate after 1 update; total wall: 53.84s
```

Measured training-path evidence:

- contrast norm `3.81629`; extraction angry probability `0.999996`; extraction speaker
  similarity `0.954005`;
- four train bases from speaker 0014 and four disjoint held-out bases from speaker 0012;
- 4/4 pass-1 rows EOS-terminated; replay/codec/experts/backward took `1.412s` after
  `16.891s` native generation;
- finite nonzero pre-clip transform gradient norm `7.79739`;
- optimizer parameter delta L2 `0.383995` and `T(v)` output delta L2 `0.193319`;
- every Qwen/talker/codec/emotion2vec/WavLM parameter remained frozen and gradient-free.

Selected learned arm: update 1, alpha `0.73`. Across the four held-out neutral texts,
mean angry probability was `0.749920` versus `0.000000637` unsteered (positive change
`+0.749919`); mean WavLM speaker similarity was `0.852267` versus `0.866093`
(degradation `0.013826`); and EOS termination was 4/4. Per-row angry probabilities
were `0.999986`, `0.999695`, `1.000000`, and `0.000000395`: three rows became strongly
angry, while the fourth improved only numerically. The strongest raw-vector control
scored `0.996263`, so this run proves that the learned transform is trainable and that
its selected output steers several unseen neutral texts toward angry; it does **not**
show that the transform beats a tuned raw contrast or works for every neutral utterance.
Here “optimal” means the best feasible learned checkpoint/alpha measured by this bounded
run, not a universal optimum.

Artifacts:

- `runs/angry-smoke-20260720T022733Z/learned_angry_transform.pt`
- `runs/angry-smoke-20260720T022733Z/angry_steering_payload.pt`
- `runs/angry-smoke-20260720T022733Z/report.json`
- `runs/angry-smoke-20260720T022733Z/audio/`

The payload audit passed: raw, transformed, and scaled vectors are finite 1024-D
tensors; reloading the transform reproduces `transformed_v`; and
`scaled_vector == transformed_v * 0.73`. WER and true ISR were not measured in this
earlier pre-Whisper proof (EOS survival is not a substitute). Before a production or
“any utterance” claim, run multiple pairs, speakers, and seeds; add the independent
WER/ISR panel and an independent emotion judge; and compare the learned transform
against the unusually strong tuned raw-vector control.

### 2026-07-20 ~04:47 UTC — isolated real Whisper gradient smoke (PASS)

An interactive GPU1 check loaded the pinned
`openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
and scored the real WAV `data/esd-smoke-source/wav/0012_000332.wav` against its
requested transcript. Model load took `2.508s`; token NLL was `1.77177894`; forward
took `0.786s`; and the waveform gradient norm was finite and nonzero at `62.9582`.
Whisper parameter-gradient count was zero, as required for a frozen expert. Greedy
decoded text was `" Where are you going?"`, and peak CUDA allocation was approximately
`2934.5 MiB`.

This isolates the contract that matters before composing the full loop: token NLL can
differentiate through frozen Whisper and its torch-native frontend back to audio
without giving gradients to Whisper parameters. It does not test the codec/STE/Qwen
portion and is not evidence that the ASR term improves WER. This was a console-only
interactive check and was not separately persisted as an artifact; the following
end-to-end smoke is the durable composed record.

### 2026-07-20 04:46:12–04:47:06 UTC — full one-pair smoke with Whisper (PASS)

This reran the complete one-contrast/four-train-base proof after adding frozen
Whisper-large-v3-turbo token NLL to the differentiable training loss and decoded WER
to the held-out gate:

```text
HF_HUB_OFFLINE=1 python -m ste_optimized smoke \
  -c configs/angry-smoke.yaml --pair-id 0011:000254 \
  --output runs/angry-smoke-whisper-20260720T0448Z
status: pass; success gate after 1 update; total wall: 54.223s
```

All four training rows survived. The update measured emotion loss `10.710823`, speaker
loss `0.263617`, and Whisper token NLL `3.023123`; the finite nonzero pre-clip transform
gradient norm was `7.821536`, parameter delta L2 `0.383995`, and steering delta L2
`0.193319`. Phase timers were `0.222s` prompt preparation, `16.796s` native Qwen
generation, `2.287s` replay/codec/all-experts/backward, and `0.029s` optimizer.
FlashAttention 2 was active in model, talker, and code predictor.

The selected held-out arm was update 1 at alpha `0.75`: mean angry probability
`0.750684`, mean speaker similarity `0.875699`, 4/4 rows angrier than unsteered, and
4/4 EOS termination. Decoded corpus WER was `0/15 = 0.0`; the **matched raw alpha-0.75
control** also had WER `0.0`, hence absolute degradation `0.0`, which passes the
inclusive `≤ +0.06` gate. This is only a four-utterance, 15-reference-word plumbing
check (`min_validation_reference_words=1` for smoke), not production WER evidence;
the production configuration requires at least 100 reference words. The learned arm
still did not beat the strongest tuned raw anger control (raw alpha `0.73` mean anger
`0.996263`), so the run proves end-to-end trainability with an ASR constraint, not
learned-transform superiority or generalization.

Artifacts are under `runs/angry-smoke-whisper-20260720T0448Z/`, including
`learned_angry_transform.pt`, `angry_steering_payload.pt`, `report.json`, per-update
checkpoint, and generated audio.

### 2026-07-20, update ending 04:49:24 UTC — B32 full update timing (PASS)

One complete update of the one-hour configuration ran with **K=16 contrasts × M=2
bases = 32 rows**, replayed in **eight chunks of 4**. All 32/32 rows EOS-terminated and
were scored by emotion2vec, WavLM, and differentiable Whisper before backward. The
exact command and recorded result were:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python -m ste_optimized train -c configs/angry-1h.yaml \
  --max-updates 1 \
  --output runs/angry-b32-whisper-smoke-20260720T0450Z

emotion loss                         7.242962
speaker loss                         0.191268
Whisper token NLL                    2.223482
mean angry probability               0.385846
mean speaker similarity              0.808732
pre-clip transform gradient norm     2.009098
parameter delta L2                   0.127998
steering delta L2                    0.082972
prompt preparation                   1.822s
pass-1 native Qwen generation       17.021s
pass-2 replay/codec/experts/backward 13.409s
optimizer                            0.027s
total recorded phase time           32.279s
```

The transform artifact and row-level log are in
`runs/angry-b32-whisper-smoke-20260720T0450Z/`. This single update verifies the B32
memory/gradient path and gives an initial timing point. It is not a convergence result:
one update cannot establish loss trend, validation improvement, repeatability, or a
one-hour end-to-end wall time.

### 2026-07-20 04:52 UTC — CPU regression suite after ASR/WER integration (PASS)

```text
pytest -q -m 'not gpu'
55 passed, 5 deselected in 4.27s
```

This is unit/regression evidence only; the CUDA results above carry the real-model
gradient and timing claims.

### 2026-07-20 04:57:35–05:57:28 UTC — bounded two-seed training and validation (PASS)

Two independent seeds ran concurrently, one per RTX A6000:

```text
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  python -m ste_optimized train -c configs/angry-1h.yaml --seed 42 \
  --output runs/angry-1h-seed42-20260720T0452Z

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  python -m ste_optimized train -c configs/angry-1h.yaml --seed 43 \
  --output runs/angry-1h-seed43-20260720T0452Z
```

Both commands completed all 50 attempted/50 successful optimizer updates with no
skips, OOMs, non-finite gradients, or frozen-model gradient violations. Run-directory
creation through final transform export took `1494.469s = 24m54.5s` for seed 42 and
`1509.422s = 25m09.4s` for seed 43, including model load and five cadence panels.
Every generated row EOS-terminated. Because the then-current sampler emitted its
remainder, each seed used 46 nominal **B32** updates and four **B2** epoch tails
(updates 11/22/33/44): `1480/1480` rows survived per seed. Full B32 phase time averaged
`28.849s` (seed 42) and `29.212s` (seed 43); B32 native generation averaged
`17.130s`/`17.149s`, and replay/codec/three-expert/backward averaged
`11.667s`/`12.014s`. The B2 tails were inefficient (`11.54s`/`11.05s` for only two
rows), motivating the carried-tail full-batch sampler fix described in §5.

Training-component trends, comparing the first ten with last ten updates:

| Seed | angry prob | emotion loss | speaker sim | Whisper token NLL |
|---|---:|---:|---:|---:|
| 42 | `0.42617 → 0.60570` | `6.54879 → 3.14914` | `0.82847 → 0.78394` | `1.76231 → 1.92292` |
| 43 | `0.36579 → 0.57767` | `6.97225 → 3.23189` | `0.82269 → 0.77114` | `1.87302 → 2.05840` |

Thus training learned a stronger anger direction, but late checkpoints traded away
speaker/ASR quality. The WER-aware checkpoint selection was necessary; the final
checkpoint was not assumed best. Pre-clip transform gradients remained finite and
positive on every update (`0.3338–15.0408` for seed 42;
`0.2409–26.4030` for seed 43). Ten periodic transforms per seed were saved. The live
cadence subset in these processes predated the speaker-interleaving fix and was not
used for the final claim; post-training selection used the corrected balanced panel.

#### Checkpoint × alpha screen (provisional)

`evaluate-checkpoints` loaded Qwen/emotion2vec/WavLM/Whisper once per GPU and scored
all 10 unique snapshots × alphas `{0.5, 0.73, 0.75, 1.0, 1.25}` against matched raw-v
controls. The provisional panel had 12 rows, alternating 6 contrasts from speaker 0012
and 6 from 0016, and 87 normalized reference words. It found 5 fully gated arms for
seed 42 and 3 for seed 43. The best provisional arms were:

- seed 42, update 45, alpha `0.75`: angry `0.90825` vs raw `0.50070`, delta
  `+0.40755`, 95% CI `[+0.15089,+0.66459]`; speaker `0.86109`; ISR `1.0`; WER
  `6/87=6.90%` vs raw `5/87=5.75%`, delta `+1.15 pp`;
- seed 43, update 30, alpha `0.73`: angry `0.83330` vs raw `0.52048`, delta
  `+0.31282`, CI `[+0.08747,+0.56313]`; speaker `0.86679`; ISR `1.0`; WER
  `1/87=1.15%` vs raw `2/87=2.30%`, delta `−1.15 pp`.

Reports:
`runs/angry-1h-seed42-20260720T0452Z/checkpoint_panel_screen.json` and
`runs/angry-1h-seed43-20260720T0452Z/checkpoint_panel_screen.json`.

#### Full 40-contrast/276-word validation

Each provisional winner was then evaluated over all 40 held-out validation contrasts
(20 each from speakers 0012 and 0016), all five alphas, matched raw-v controls, and
deterministic generation chunks of at most B32. The outcome was:

| Candidate | alpha | angry learned/raw | delta (95% CI) | speaker learned/raw | ISR | WER learned/raw | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| seed 42, update 45 | **0.73** | `0.72354 / 0.46203` | `+0.26152 [0.07032, 0.43689]` | `0.85566 / 0.86317` | `1.0 / 1.0` | `23/276 (8.33%) / 23/276 (8.33%)` | **all gates PASS** |
| seed 43, update 30 | 0.73 | `0.72385 / 0.46203` | `+0.26182 [0.06649, 0.44807]` | `0.84475 / 0.86317` | `1.0 / 1.0` | `26/276 (9.42%) / 23/276 (8.33%)` | FAIL speaker absolute gate by `0.00525` |

For the accepted seed-42 arm, the exact WER counts were learned `S=13,D=9,I=1`
and raw `S=13,D=10,I=0`: both have 23 errors, so WER degradation is exactly `0.0`,
well inside the inclusive `+0.06` contract. All 40 pairs were scored and terminated;
learned steering raised angry probability over its matched raw control on **29/40
(72.5%)** rows. This is the checkpoint "accuracy"/directionality rate; it supports an
aggregate held-out generalization claim but explicitly does not establish the stated
aspiration of improvement for every pair.

Alpha is a real inference hyperparameter, not a harmless rescaling. For the same
accepted seed-42 checkpoint, alpha `0.75` produced WER `471/276=170.65%` versus its
raw control `20/276=7.25%`, failing emotion-CI, speaker, and WER gates, even though
alpha `0.73` passed all of them. Autoregressive sampling is discontinuous enough that
the selected `0.73` should be locked for this checkpoint; nearby alphas require their
own WER validation.

Accepted artifacts:

- source checkpoint:
  `runs/angry-1h-seed42-20260720T0452Z/checkpoints/transform-00045.pt`;
- portable selected copy:
  `runs/angry-full-seed42-u45-20260720T0549Z/final_transform.pt`;
- five-alpha full report:
  `runs/angry-full-seed42-u45-20260720T0549Z/checkpoint_panel_full.json`;
- selected-alpha report with 29/40 directionality:
  `runs/angry-full-seed42-u45-20260720T0549Z/checkpoint_panel_selected_alpha073.json`.

The source and selected transform files have identical SHA-256
`77ea3721a8e5cf08f50cb481dca6ecdfe3cf9acc6cb12959036db593b7c450df`.
The second-seed full report is
`runs/angry-full-seed43-u30-20260720T0549Z/checkpoint_panel_full.json`.

Measured from the earliest seed run-directory creation (`04:57:35.984 UTC`) through
the final selected-alpha report (`05:57:28.721 UTC`), the complete two-seed training,
exhaustive checkpoint/alpha screen, full validation, and directionality audit took
**3592.737s = 59m52.7s**. This includes roughly ten idle minutes waiting for job-launch
authorization, so the computational pipeline itself has additional headroom.

### 2026-07-20 05:58 UTC — final CPU regression suite (PASS)

```text
PYTHONPATH=src pytest -q -m 'not gpu'
63 passed, 5 deselected in 4.63s
```

This includes ASR/WER, memory-bounded checkpoint evaluation, speaker-balanced panels,
carried-tail full batching/resume, and fail-closed cadence-best selection.

## 9. Known limits

Fixed-hard STE drops cross-timestep credit and the EOS/duration path (duration is a
first-order emotion cue — an objective ceiling, not a bug); emotion2vec is both trainer
and judge, so a positive result is provisional until independently checked. Likewise,
the differentiable transcript loss and decoded WER both use the pinned Whisper model;
the matched-raw WER gate is useful preservation evidence but not an independent ASR
validation. The completed validation used one deterministic generation seed, two
held-out speakers, and 40 contrast rows; 29/40 improved, not 40/40. The test split is
still untouched. The result therefore supports the bounded aggregate gate and the
one-hour systems goal, but not a universal "any contrast pair" or independently judged
emotion claim. Alpha sensitivity was extreme around 0.73–0.75 and must remain gated by
WER. Batching changes the cost of the experiment, not its epistemics.
