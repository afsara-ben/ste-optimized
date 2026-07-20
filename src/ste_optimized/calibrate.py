"""Machine calibration — run BEFORE quoting any schedule (plan §5).

Five micro-benchmarks; writes calibration.json with measured rates and
recommended settings. Reference values (RTX A6000-48GB, bf16, flash-attn 2)
are in the portable plan; never reuse them unmeasured on another machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .backend import QwenTTSBackend
from .config import ExperimentConfig
from .data import load_bases


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_calibration(cfg: ExperimentConfig, out_path: str | Path,
                    gen_batches: tuple[int, ...] = (1, 8, 16, 32)) -> dict:
    report: dict = {"device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu"}
    backend = QwenTTSBackend(cfg.model)
    report["model_load_seconds"] = round(backend.load_seconds, 2)
    talker = backend.talker
    H = cfg.model.hidden_size
    dev = backend.device

    # --- 1+2. generation: single-stream + batched scaling -------------------
    bases = load_bases(cfg.data.dataset_dir, "train")
    if bases:
        rows = [{"base_id": b.base_id, "target_text": b.target_text,
                 "reference_text": b.reference_text,
                 "reference_audio": b.reference_audio}
                for b in (bases * ((max(gen_batches) // len(bases)) + 1))]
        gen_report = {}
        backend.generate_prepared_batch(  # warmup
            backend.prepare_voice_clone_prompts(rows[:1]), None, cfg.sampling, 0)
        for B in gen_batches:
            entries = backend.prepare_voice_clone_prompts(rows[:B])
            t0 = time.perf_counter()
            g = backend.generate_prepared_batch(entries, None, cfg.sampling, 42)
            _sync()
            wall = time.perf_counter() - t0
            gen_report[str(B)] = {
                "frames_per_s": round(sum(g.lengths) / wall, 1),
                "s_per_step": round(wall / max(g.lengths), 4)}
        report["generation"] = gen_report
    else:
        report["generation"] = "skipped: no bases (run build-data first)"

    # --- 3. replay-style teacher-forced fwd+bwd -----------------------------
    tf_report = {}
    for B, S, T in ((8, 192, 64), (16, 192, 64), (32, 192, 64)):
        try:
            steer = torch.zeros(H, device=dev, dtype=torch.float32,
                                requires_grad=True)
            embeds = torch.randn(B, S, H, device=dev, dtype=torch.bfloat16)
            mask = torch.ones(B, S, dtype=torch.long, device=dev)
            labels = torch.randint(0, 2000, (B, S), device=dev)
            labels[:, : S - T] = -100
            codes = torch.randint(0, 2000, (B * T, 16), device=dev)

            def step():
                e = embeds.clone()
                e[:, S - T:, :] += steer.to(torch.bfloat16)
                out = talker(inputs_embeds=e, attention_mask=mask,
                             labels=labels, use_cache=False,
                             output_hidden_states=True)
                hs = out.hidden_states[0] if isinstance(out.hidden_states, tuple) \
                    else out.hidden_states
                hid = hs[-1][:, S - T:, :].reshape(B * T, H)
                _, sub = talker.forward_sub_talker_finetune(codes, hid)
                (out.loss + 0.3 * sub).backward()
                steer.grad = None

            step(); step(); _sync()
            t0 = time.perf_counter(); step(); _sync()
            tf_report[f"B{B}"] = {"ms": round((time.perf_counter() - t0) * 1e3),
                                  "peak_gb": round(
                                      torch.cuda.max_memory_allocated() / 1e9, 1)}
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            tf_report[f"B{B}"] = "OOM"
            torch.cuda.empty_cache()
    report["tf_fwd_bwd"] = tf_report

    # --- 4. codec decode + experts (forward, batched) -----------------------
    st = backend.tts.model.speech_tokenizer
    codes8 = [torch.randint(0, 2048, (100, 16)) for _ in range(8)]
    st.decode({"audio_codes": [codes8[0]]}); _sync()
    t0 = time.perf_counter()
    wavs, sr = st.decode({"audio_codes": codes8}); _sync()
    report["codec_decode_ms_per_row_b8"] = round(
        (time.perf_counter() - t0) / 8 * 1e3, 1)
    try:
        from .experts import ExpertSuite
        experts = ExpertSuite.load(cfg.model.device)
        waves = [torch.randn(16000 * 3, device=dev) for _ in range(8)]
        experts.emotion.logits(waves); _sync()
        t0 = time.perf_counter(); experts.emotion.logits(waves); _sync()
        report["emotion2vec_ms_per_row_b8"] = round(
            (time.perf_counter() - t0) / 8 * 1e3, 1)
        experts.speaker.embed(waves); _sync()
        t0 = time.perf_counter(); experts.speaker.embed(waves); _sync()
        report["wavlm_ms_per_row_b8"] = round(
            (time.perf_counter() - t0) / 8 * 1e3, 1)
    except Exception as exc:  # experts extra not installed
        report["experts"] = f"skipped: {exc}"

    # --- 5. reminder --------------------------------------------------------
    report["note"] = ("Benchmark 5 (full-loop 10-update timed smoke) = "
                      "`ste-optimized train --max-updates 10`; it is the only "
                      "trustworthy source for the pass-2 backward cost.")
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report
