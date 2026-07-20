# Batched Expert-in-the-Loop Replay Training — Portable Plan

**Created:** 2026-07-19 (UTC)
**Updated:** 2026-07-20 02:28 UTC — **INITIAL REAL-CONTRAST CUDA SMOKE PASSED;
PRODUCTION VALIDATION PENDING.** The plan below is now code: a
self-contained package at `ste-optimized/` (repo root; own `pyproject.toml`, package
`ste_optimized`, no dependency on the `emotionsteer` package — designed to be moved
into its own git repo). Status: the bounded `smoke` workflow is implemented; 37 CPU
tests pass; three real-WAV GPU parity components pass; and one real angry-minus-neutral
contrast completed the full FlashAttention Qwen → replay → reference-conditioned codec
→ emotion2vec/WavLM → backward path and exported a learned transform and steering
payload. This establishes the initial plumbing/learning proof, not the production claim
that the vector generalizes to every neutral utterance; §8 records the exact evidence
and limitations. Section 6 gives the commands and maps each plan phase to its module.
§3.1 contains the exact loss formula, two-pass T(v) recomputation, and gradient-path
semantics including what STE captures versus drops.
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
**Derived from:** `BATCHED_REPLAY_PLAN_july19.md` (machine-specific measurements and
file-level work items live there; this document is the machine-independent, executable
version). Objective and estimator are those of the original replay program; only the
batching design and training hygiene change.

---

## 1. Goal

Train a rank-16 low-rank transform `T_θ(v) = v + U(Dv)` (identity init: orthonormal
random `D`, zero `U`; 32,768 params) mapping a layer-15 contrast `v ∈ R^1024` to an
improved steering vector `u`, by **generating steered audio during training and
backpropagating frozen expert losses** (emotion2vec + WavLM) through a fixed-hard-STE
replay graph into `T_θ` only. Qwen and the experts stay frozen; no RL; no reward models.

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
  (validation WER only).
- **Data per emotion:** ~200 train pair contrasts + neutral-base catalog, ≥18
  (target 40) validation contrasts, extracted with the frozen extraction pipeline;
  per-pair extraction-time expert scores retained as fixed loss weights. Canonical
  speaker partition (mandatory): train 0011/0014/0017/0020, val 0012/0016,
  test 0013/0019, reserve 0015/0018.

## 3. The batched update (Algorithm)

One optimizer update = `K` contrasts × `M` bases = `R` rows (defaults `K=10, M=4,
R=40`; rationale in §4).

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

PASS 2 — gradients, in chunks of ~8 rows with gradient accumulation:
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
     → waveforms → batched emotion2vec loss + WavLM speaker loss
  9  chunk loss = Σ rows [emotion_loss + speaker_weight · speaker_loss] · w_row;
     backward per chunk (accumulate)

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
        w_i · [ −log P_e2v(c | wav_{i,b})                     # emotion loss
                + λ_spk · (1 − cos(xvec(wav_{i,b}), xvec(ref_b))) ]   # speaker loss
    + λ_id·L_identity + λ_cos·L_cosine + λ_norm·L_norm        # keep T(v) near v
```

The regularizers, averaged over the update's K contrast vectors (fp32; their
gradient reaches θ directly, no model in the path), with d = hidden size (1024):

```text
L_identity = mean_i  ‖T_θ(v_i) − v_i‖² / d          # small residual
L_cosine   = mean_i  ( 1 − cos(T_θ(v_i), v_i) )     # keep direction
L_norm     = mean_i  ( log( ‖T_θ(v_i)‖ / ‖v_i‖ ) )² # keep norm ratio near 1
```

Defaults: λ_spk = 1.0; λ_id = λ_cos = λ_norm = 0.01; N = total rows of the update
(across ranks if sharded); w_i = the pair's extraction-time target-emotion
probability — a CONSTANT, no expert gradients flow through it.
P_e2v = frozen emotion2vec head; xvec = frozen WavLM-SV embedding; ref_b = the base's
ICL reference recording.

Gradient path (only θ trains): L → experts → waveform → codec decoder → ŷ
(∂ŷ/∂logits = ∂p/∂logits) → replay logits → talker layers above 15 → the
renormalised steering injection at layer 15 → u_i → θ, plus the direct regularizer
path. Every module on the way is frozen but differentiable.

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
  evaluations of one input. Prefer diversity: **K=10 × M=4** default, floor `M=2`
  (per-contrast diagnostics get too noisy below), `M=5` only as a
  historical-comparability arm. Bases are resampled each epoch, so coverage of each
  contrast's ~15 eligible cross-speaker bases accumulates within a few epochs anyway.

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
4. Batched codec decode + emotion2vec + WavLM forward per row
   [25 / 8 / 3 ms/row at B=8].
5. **10-update timed smoke** of the full loop — the only trustworthy source for the
   pass-2 backward (differentiable codec + experts, checkpointed) [single-row
   reference ~0.36 s at 50 frames; batched value must be measured, ±2× a priori].

Schedule formulas:

```text
t_update ≈ R × mean_frames / gen_frames_per_s   (pass 1)
         + n_chunks × t_chunk(measured in #5)   (pass 2)
         + ~1 s                                 (cached prompts, optimizer, logs)
epoch    = ceil(N_train / K) updates
training ≤ 300 updates with early stop (validate every 25, patience 5)
```

Reference outcome (A6000): t_update ≈ 30–45 s for K=10 ⇒ epoch ~10–16 min ⇒ full
training ~2.5–4 h/seed ⇒ one emotion end-to-end ~3.5–5.5 h wall with 2 seeds parallel.

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
| ESD ingest, pair/base records, fail-closed leakage checks | `data.py` |
| Batched resumable mean-decode contrast extraction + fixed weights (nothing pre-extracted assumed) | `extraction.py` |
| K×M sampler, per-epoch base rotation, resume | `sampling.py` |
| Update loop: survival policy, chunked accumulation, AdamW+warmup/cosine, phase timings, checkpoints, early stop | `training.py` |
| Cadence + full gated panel, cached controls, bootstrap CI | `evaluation.py` |
| Micro-benchmarks (§5) | `calibrate.py` |
| Seed-parallel policy + optional torchrun `ddp_rows` row-sharding | `distributed.py` |

- **Stage 0 — initial real-contrast acceptance COMPLETE; production acceptance
  pending.** The `smoke` command uses one extracted angry-minus-neutral contrast,
  broadcasts the same `T(v)` across four neutral training rows, replays in two-row
  chunks, sends decoded audio through both frozen experts, rejects zero/non-finite
  transform gradients, and evaluates the learned vector on four disjoint neutral
  utterances. It is attempt/wall bounded and exports only after its held-out gates
  pass. The 2026-07-20 run passed on an RTX A6000 with FlashAttention 2 (§8).
  Prompt assembly, bare/reference soft-codec, and emotion2vec parity also passed on
  a real WAV. DDP (`ddp_rows`), length bucketing, other emotions, repeated seeds,
  independent emotion judging, WER/true ISR, and the full production panel remain
  deferred; native processed-logit parity for the replay approximation is also not
  yet a dedicated GPU assertion.
- **Stage 1 — data:**
  `ste-optimized build-data -c configs/angry.yaml --source /path/to/ESD` then
  `ste-optimized extract -c configs/angry.yaml --split train` (and `validation`);
  set `data.contrasts_path`. Extraction is batched (~25× over scalar) and resumable.
- **Stage 2 — gates:** the milestone gates above (acceptance → multi-chunk →
  parity, in order) + `calibrate` + the 10-update timed smoke (§5) — the smoke
  is the only trustworthy source for the pass-2 backward cost.
- **Stage 3 — train:** 60-update pilot → inspect cadence panel → ≤300 updates with
  early stopping; two seeds, one GPU each:
  `CUDA_VISIBLE_DEVICES=0 ste-optimized train -c configs/angry.yaml --seed 42 --output runs/s42 &`
  `CUDA_VISIBLE_DEVICES=1 ste-optimized train -c configs/angry.yaml --seed 43 --output runs/s43 &`
  (optional row-sharding instead: `torchrun --standalone --nproc-per-node=2 -m
  ste_optimized train -c … --distributed ddp_rows`; historically ~1.13× — seed-parallel
  is the recommended use of extra GPUs.)
- **Stage 4 — validation gate** (unchanged from the program):
  `ste-optimized evaluate -c configs/angry.yaml --transform runs/s42/best_transform.pt`
  — fixed panel, batched generation, controls cached once; per-pair T(v) vs raw v at
  α=1; exported T(v_global) and raw v_global over the alpha sweep; closed-form
  purifiers; zero-input control. Gates: emotion-prob delta > 0 with 95%
  paired-bootstrap CI excluding 0; WavLM ≥ 0.85 and degradation ≤ 0.05; ISR ≥ 80% and
  ≥ control − 10 pp; WER ≤ control + 5 pp. Checkpoint chosen on validation only; test
  split touched once, after freezing.
- **Stage 5 — fallback:** teacher-forced likelihood-preference trainer
  (`TEACHER_FORCED_PREFERENCE_PLAN_PORTABLE.md`) if replay fails gates.
- **Stage 6 — replicate** per emotion; shared conditioned transform only after all pass.

## 7. Verification checklist (implemented in `ste-optimized/tests/`)

- **CPU unit tests (37, green):** `test_transform.py` (identity init, T(0)=0,
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
  regressions in `test_training_bounds.py`, `test_codec.py`, and `test_smoke.py`.
- **GPU milestone gates (5, `pytest -m gpu`, in order):** the one-fixed-batch
  acceptance test (complete reference-conditioned loss; four assertions),
  multi-chunk accumulation, prompt-assembly parity, soft-codec parity (bare
  and reference-conditioned), emotion2vec-head parity.
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
short proof (EOS survival is not a substitute). Before a production or “any utterance”
claim, run multiple pairs, speakers, and seeds; add the independent WER/ISR panel and
an independent emotion judge; and compare the learned transform against the unusually
strong tuned raw-vector control.

## 9. Known limits

Fixed-hard STE drops cross-timestep credit and the EOS/duration path (duration is a
first-order emotion cue — an objective ceiling, not a bug); emotion2vec is both trainer
and judge, so a positive result is provisional until independently checked. Batching
changes the cost of the experiment, not its epistemics.
