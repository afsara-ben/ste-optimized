# ste-optimized

Batched **expert-in-the-loop** replay training of a rank-16 steering-vector
transform for `Qwen/Qwen3-TTS-12Hz-0.6B-Base` emotion steering. Self-contained:
no dependency on any other project package; drop this folder into its own git
repo. Design and rationale: `BATCHED_REPLAY_PLAN_PORTABLE.md` (copy it into
this repo alongside this README).

**What trains:** only a 32,768-parameter low-rank transform `T(v) = v + U(Dv)`.
**What supervises it:** steering is applied *during generation*; the sampled
codec tokens are replayed teacher-forced with fixed-hard STE; the waveform is
decoded differentiably; frozen **emotion2vec + WavLM** score the audio and the
loss backpropagates through the whole replay graph into `T` only.
**Why it's fast:** every stage is batched. The Qwen decode loop is
overhead-bound, so batched generation gives ~25x at batch 32 (measured,
A6000); codec/expert scoring costs ~50-60 ms/row batched.

## Requirements

- 1 CUDA GPU, >= 24 GB VRAM (48 GB comfortable). Second GPU optional
  (seed-parallel; optional `ddp_rows` row-sharding under torchrun).
- Python >= 3.11, then:

```bash
pip install -e ".[experts,eval,dev]"
```

- Data: ESD-style corpus — per speaker, neutral + emotional renditions of the
  same texts (official ESD layout or a CSV `path,speaker,emotion,text,index`).
- Models download automatically on first use (Qwen3-TTS 0.6B @ pinned
  revision, emotion2vec_plus_large, wavlm-base-plus-sv; ~5 GB total).

## Pipeline (nothing pre-extracted is assumed)

```bash
# 1. Pair audio, choose neutral bases, write hashed manifest + leakage checks
ste-optimized build-data -c configs/angry.yaml --source /path/to/ESD

# 2. Batched mean-decode contrast extraction (train, then validation)
ste-optimized extract -c configs/angry.yaml --split train
ste-optimized extract -c configs/angry.yaml --split validation
#    -> set data.contrasts_path in the config to the printed train path

# 3. Machine calibration (REQUIRED before trusting any schedule)
ste-optimized calibrate -c configs/angry.yaml

# 4. GPU gate tests + 10-update timed smoke (pins the pass-2 backward cost)
pytest -m gpu tests/
ste-optimized train -c configs/angry.yaml --max-updates 10

# 5. Train (2 seeds; one GPU each = recommended multi-GPU use)
CUDA_VISIBLE_DEVICES=0 ste-optimized train -c configs/angry.yaml --seed 42 --output runs/s42 &
CUDA_VISIBLE_DEVICES=1 ste-optimized train -c configs/angry.yaml --seed 43 --output runs/s43 &

# optional instead: shard each update's rows across GPUs (measured ~1.13x only)
torchrun --standalone --nproc-per-node=2 -m ste_optimized train \
    -c configs/angry.yaml --distributed ddp_rows

# 6. Full gated validation panel
ste-optimized evaluate -c configs/angry.yaml --transform runs/s42/best_transform.pt
```

## Module map

| Module | Role |
|---|---|
| `config.py` | YAML -> dataclasses; canonical speaker partition; provenance fingerprint |
| `backend.py` | model load, prompt cache by base_id, **native batched generation** with per-row steering, replay prompt assembly |
| `hooks.py` | layer-15 steering hooks (decode-step / masked-replay) with per-position norm restoration; activation capture for extraction |
| `ste.py` | fixed-hard STE + vectorised replay of the pass-1 logit chain (repetition penalty, suppress, min-new-tokens, temp/top-k/top-p) |
| `replay.py` | pass-2 padded teacher-forced forward, target gathering, subtalker scoring, STE one-hot assembly |
| `codec.py` | differentiable soft-code decode: STE one-hots -> frozen decoder (chunked + checkpointed) |
| `experts.py` | frozen emotion2vec + WavLM, batched, differentiable; parity check vs funasr inference |
| `data.py` | ESD ingest, pair/base records, fail-closed leakage checks |
| `extraction.py` | batched, resumable mean-decode contrast extraction + fixed pair weights |
| `sampling.py` | K contrasts x M bases per update, per-epoch base rotation, resumable |
| `training.py` | the batched update loop: pass 1 -> chunked pass 2 -> AdamW step; survival policy; JSONL phase timings; checkpoints |
| `evaluation.py` | cadence panel + full gated panel with cached controls, bootstrap CI gates |
| `calibrate.py` | the 5 machine micro-benchmarks |
| `distributed.py` | seed-parallel policy + optional torchrun row-sharding |

## Initial milestone (simplified, 2026-07-20)

**One angry transform. One fixed batch. One acceptance test.** Everything else
waits until it passes:

```bash
STE_OPT_REF_WAV=/path/to/speech.wav pytest -m gpu tests/test_parity.py
```

1. **`test_acceptance_one_fixed_batch` — THE gate.** The complete
   reference-conditioned waveform loss (generate → replay STE → codec decode
   with the reference-code prefix + trim, exactly as native inference decodes
   → emotion2vec + WavLM) must: produce a finite, nonzero expert-only gradient
   in T_θ; change T_θ's parameters on an optimizer step; decrease under
   repeated optimization on the same fixed batch; and leave every frozen model
   (talker, codec, emotion2vec, WavLM) without parameter gradients.
2. **`test_multichunk_accumulation`** — two chunks with per-chunk T(v)
   recompute: no freed-graph error, gradients accumulate, frozen models clean.
3. **Native/replay parity** — prompt assembly vs captured native prefill;
   soft-vs-hard codec (bare AND reference-conditioned); emotion2vec head vs
   funasr inference.

**Deferred until all of the above pass:** `ddp_rows` distributed mode, length
bucketing, happy/sad emotions, the full gated validation campaign.

CPU unit tests (`pytest` with no marker) cover the transform, STE math, hook
renormalisation (incl. the `steer_frame0_predictor` flag), the multi-chunk
freed-graph regression, sampler balance/rotation/resume, and leakage
fail-closed behaviour.

## Correctness notes (fixed 2026-07-20 after external review)

- Pass-2 recomputes `T(v)` **per chunk** (a shared forward across chunk
  backwards hits freed autograd buffers — pinned by
  `tests/test_training_chunks.py`).
- The loss/eval decode is **reference-conditioned**: `cat([ref_codes,
  generated])` then a proportional trim, mirroring native inference
  (`qwen3_tts_model.py:614-629`) so experts score the waveform real Qwen
  output produces.
- STE one-hots are cast to the codec's parameter dtype (fp32 → bf16 conv
  would crash); the speech tokenizer is frozen explicitly (its plain-class
  wrapper hides it from `tts.model.parameters()`), and the gradient-contract
  tripwire covers talker + codec + both experts.
- Residual codebooks (1–15) replay the subtalker's own sampling warpers
  (temperature 0.9 / top-k 50 by default) before the STE softmax.
- `model.steer_frame0_predictor` (default off) optionally steers the last
  prompt position — the frame-0 predictor — symmetrically in generation,
  replay, and evaluation.

## Known limits (by design; see the plan)

Fixed-hard STE drops cross-timestep credit and the EOS/duration gradient path
(duration is a first-order emotion cue — an objective ceiling). emotion2vec is
both trainer and judge, so positive results are provisional until checked by an
independent classifier or listening test. Batched sampling does not reproduce
single-stream RNG token streams; reproducibility is per ordered batch identity.
