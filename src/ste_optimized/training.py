"""The batched expert-in-the-loop training loop (plan §3).

One optimizer update:
  1. PASS 1 — one batched no-grad generation of K x M rows, each row steered by
     its contrast's detached u_i = T(v_i); survival policy >= min_row_survival.
  2. PASS 2 — chunks of `chunk_rows` with gradient accumulation:
     replay forward (+STE) -> differentiable codec -> expert losses -> backward.
  3. clip, AdamW step (warmup+cosine), per-phase timings to training.jsonl.

Distributed:
  - mode "seed_parallel": nothing here — launch N independent processes.
  - mode "ddp_rows": under torchrun, rank r keeps rows with index % world == r
    in pass 1/2 and transform gradients are all-reduced (mean) before the step.
    Kept optional; measured historically at only ~1.13x.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .backend import QwenTTSBackend
from .codec import decode_soft, output_sample_rate
from .config import ExperimentConfig
from .data import load_bases
from .distributed import DistributedContext
from .experts import ExpertSuite, resample_to_expert
from .extraction import load_contrasts
from .replay import replay_chunk
from .sampling import BatchSampler, UpdateBatch
from .transform import LowRankTransform, save_transform


def _lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        dist: DistributedContext,
        backend: QwenTTSBackend | None = None,
        experts: ExpertSuite | None = None,
        contrasts: list[dict] | None = None,
        bases: list | None = None,
    ) -> None:
        """Build a trainer, optionally reusing already-loaded resources.

        The optional arguments make an extract -> smoke-train process reuse the
        large frozen models and allow a one-pair in-memory smoke fixture.  The
        original ``Trainer(cfg, dist)`` API and on-disk loading remain the
        default production path.
        """
        self.cfg = cfg
        self.dist = dist
        self.out = Path(cfg.train.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "training.jsonl"

        self.backend = backend if backend is not None else QwenTTSBackend(cfg.model)
        self.experts = experts if experts is not None else ExpertSuite.load(cfg.model.device)
        self.sr = output_sample_rate(self.backend.tts.model.speech_tokenizer)

        if contrasts is None:
            contrasts = load_contrasts(cfg.data.contrasts_path)
        if bases is None:
            bases = load_bases(cfg.data.dataset_dir, "train")
        self.sampler = BatchSampler(
            contrasts, bases, cfg.train.contrasts_per_update,
            cfg.train.bases_per_contrast, cfg.train.seed,
            cfg.data.min_pair_emotion_prob, cfg.data.min_pair_speaker_sim)

        self.transform = LowRankTransform(cfg.model.hidden_size, cfg.train.rank)
        self.transform.initialize_identity(cfg.train.seed + 1_000_003)
        self.transform.to(self.backend.device)
        self.optimizer = torch.optim.AdamW(
            self.transform.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay)
        warmup = max(1, int(cfg.train.max_updates * cfg.train.warmup_fraction))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: _lr_lambda(s, cfg.train.max_updates, warmup))
        self.completed = 0
        self.attempts = 0
        self.best_metric = -float("inf")
        self.evals_without_improvement = 0
        self._maybe_resume()

    # ------------------------------------------------------------------ update
    def run_update(self) -> dict:
        cfg = self.cfg
        batch = self.sampler.next_batch()
        timings: dict[str, float] = {}

        # ---- pass 1: batched steered sampling (no grad) --------------------
        t0 = time.perf_counter()
        with torch.no_grad():
            u = self.transform(batch.vectors.to(self.backend.device))
        row_vectors = u[torch.tensor(batch.row_contrast, device=u.device)]
        rows = self.dist.shard_rows(batch.rows)
        row_idx = self.dist.shard_rows(list(range(len(batch.rows))))
        entries = self.backend.prepare_voice_clone_prompts(rows)
        timings["prompt_prep"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gen = self.backend.generate_prepared_batch(
            entries, row_vectors[torch.tensor(row_idx, device=u.device)].detach(),
            cfg.sampling, seed=self._batch_seed(batch), alpha=cfg.train.alpha)
        timings["pass1_generate"] = time.perf_counter() - t0

        survivors = [i for i, ok in enumerate(gen.terminated) if ok]
        survival = self.dist.global_fraction(len(survivors), len(rows))
        if survival < cfg.train.min_row_survival:
            self.optimizer.zero_grad(set_to_none=True)
            self._log({"event": "update_skipped", "survival": survival,
                       "update": self.completed, **timings})
            return {"skipped": True}

        # ---- pass 2: chunked replay -> codec -> experts -> backward --------
        self.optimizer.zero_grad(set_to_none=True)
        emotion_losses, speaker_losses, probs, sims = [], [], [], []
        t0 = time.perf_counter()
        n_scored = 0
        for cstart in range(0, len(survivors), cfg.train.chunk_rows):
            chunk_ids = survivors[cstart:cstart + cfg.train.chunk_rows]
            chunk_entries = [entries[i] for i in chunk_ids]
            chunk_codes = [gen.codes[i] for i in chunk_ids]
            chunk_slot = [batch.row_contrast[row_idx[i]] for i in chunk_ids]
            # T(v) is recomputed FRESH per chunk: each chunk's backward frees
            # the graph it traverses, so a transform forward shared across
            # chunks would make the second backward hit freed buffers.
            chunk_u = self.transform(
                batch.vectors[torch.tensor(chunk_slot)].to(self.backend.device))
            chunk_w = torch.tensor([batch.weights[s] for s in chunk_slot],
                                   device=chunk_u.device)

            rep = replay_chunk(self.backend, chunk_entries, chunk_codes,
                               chunk_u, cfg.sampling, cfg.train.alpha)
            # reference-conditioned decode: native inference decodes
            # cat([ref_code, generated]) and trims — the experts must score
            # the same waveform normal Qwen output produces.
            wavs24 = [decode_soft(self.backend.tts.model.speech_tokenizer.model,
                                  oh.unsqueeze(0), ref_codes=e.ref_code)[0]
                      for oh, e in zip(rep.ste_onehots, chunk_entries)]
            wavs16 = [resample_to_expert(w, self.sr) for w in wavs24]
            e_loss, e_prob = self.experts.emotion.loss(wavs16, cfg.data.emotion)
            ref_paths = [chunk_entries[j].reference_audio
                         for j in range(len(chunk_entries))]
            s_loss, s_sim = self.experts.speaker.loss(wavs16, ref_paths)

            loss_rows = chunk_w * (e_loss + cfg.train.speaker_weight * s_loss)
            total_rows = max(len(survivors), 1) * self.dist.world_size
            chunk_loss = loss_rows.sum() / total_rows
            reg = self.transform.regularization(
                batch.vectors.to(self.backend.device),
                cfg.train.reg_identity, cfg.train.reg_cosine, cfg.train.reg_norm
            ) * (len(chunk_ids) / max(len(survivors), 1))
            (chunk_loss + reg).backward()

            emotion_losses += e_loss.detach().tolist()
            speaker_losses += s_loss.detach().tolist()
            probs += e_prob.tolist()
            sims += s_sim.tolist()
            n_scored += len(chunk_ids)
        timings["pass2_replay_codec_experts_backward"] = time.perf_counter() - t0

        # ---- step ----------------------------------------------------------
        t0 = time.perf_counter()
        self.dist.allreduce_grads(self.transform)
        self._assert_gradient_contract()
        # Keep before/after snapshots tiny (T has only 2 * hidden * rank
        # parameters) so a smoke artifact proves that an optimizer update
        # changed both the transform and its output for the sampled contrast.
        params_before = [p.detach().clone() for p in self.transform.parameters()]
        u_before = u.detach()
        grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
            self.transform.parameters(), cfg.train.grad_clip,
            error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        with torch.no_grad():
            parameter_delta_l2 = math.sqrt(sum(
                (p.detach().float() - before.float()).pow(2).sum().item()
                for p, before in zip(self.transform.parameters(), params_before)
            ))
            u_after = self.transform(batch.vectors.to(self.backend.device))
            steering_delta_l2 = float(
                (u_after.float() - u_before.float()).norm().item())
        timings["optimizer"] = time.perf_counter() - t0
        self.completed += 1

        record = {
            "event": "update", "update": self.completed,
            "epoch": batch.epoch, "contrasts": batch.contrast_ids,
            "rows": len(rows), "survivors": n_scored, "survival": survival,
            "emotion_loss": _mean(emotion_losses),
            "speaker_loss": _mean(speaker_losses),
            "emotion_prob": _mean(probs), "speaker_sim": _mean(sims),
            # ``grad_norm`` is retained for artifact compatibility and is the
            # pre-clipping norm returned by clip_grad_norm_.
            "grad_norm": float(grad_norm_preclip),
            "grad_norm_preclip": float(grad_norm_preclip),
            "parameter_delta_l2": parameter_delta_l2,
            "steering_delta_l2": steering_delta_l2,
            "lr": self.scheduler.get_last_lr()[0],
            **{f"t_{k}": round(v, 3) for k, v in timings.items()},
        }
        self._log(record)
        if self.completed % self.cfg.train.checkpoint_every == 0:
            self._checkpoint()
        return record

    # ------------------------------------------------------------------ loop
    def train(self, cadence_eval=None) -> None:
        started = time.perf_counter()
        bounded_reason = None
        while self.completed < self.cfg.train.max_updates:
            bounded_reason = self._bounded_stop_reason(started)
            if bounded_reason is not None:
                break

            completed_before = self.completed
            self.attempts += 1
            self.run_update()
            advanced = self.completed > completed_before

            # Check the wall clock after an indivisible native generation too;
            # do not start a potentially expensive cadence panel once bounded.
            if self.cfg.train.max_wall_seconds is not None and \
                    time.perf_counter() - started >= \
                    self.cfg.train.max_wall_seconds:
                bounded_reason = "max_wall_seconds"
                break

            # A skipped update leaves ``completed`` unchanged.  Re-evaluating
            # the same cadence checkpoint on every skip is both wrong and very
            # expensive, especially when completed == 0.
            if advanced and cadence_eval is not None and \
                    self.completed % self.cfg.train.eval_every == 0 and \
                    self.dist.is_main:
                metric = cadence_eval(self)
                improved = metric > self.best_metric
                self._log({"event": "cadence_eval", "update": self.completed,
                           "metric": metric, "improved": improved})
                if improved:
                    self.best_metric = metric
                    self.evals_without_improvement = 0
                    self._export(self.out / "best_transform.pt")
                else:
                    self.evals_without_improvement += 1
                    if self.evals_without_improvement >= \
                            self.cfg.train.early_stop_patience:
                        self._log({"event": "early_stop",
                                   "update": self.completed})
                        break
        if bounded_reason is not None:
            self._log({"event": "bounded_stop", "reason": bounded_reason,
                       "update": self.completed, "attempts": self.attempts,
                       "wall_seconds": round(time.perf_counter() - started, 3)})
        if self.dist.is_main:
            self._export(self.out / "final_transform.pt")

    # ----------------------------------------------------------------- utils
    def _assert_gradient_contract(self) -> None:
        for p in self.transform.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("transform gradient missing or non-finite")
        # tripwire on EVERY frozen model in the loss path — the codec lives
        # outside tts.model.parameters() (plain-class wrapper), so it must be
        # checked explicitly, as must both experts.
        st = getattr(self.backend.tts.model, "speech_tokenizer", None)
        frozen = {
            "talker": self.backend.talker,
            "codec": getattr(st, "model", None),
            "emotion2vec": self.experts.emotion.model,
            "wavlm": self.experts.speaker.model,
        }
        for name, module in frozen.items():
            if module is None:
                continue
            for parameter_index, p in enumerate(module.parameters()):
                if p.grad is not None:
                    raise RuntimeError(
                        f"frozen model {name!r} parameter {parameter_index} "
                        "received a gradient")
                if p.requires_grad:
                    raise RuntimeError(
                        f"frozen model {name!r} parameter {parameter_index} "
                        "has requires_grad=True")

    def _bounded_stop_reason(self, started: float) -> str | None:
        train = self.cfg.train
        if train.max_attempts is not None and self.attempts >= train.max_attempts:
            return "max_attempts"
        if train.max_wall_seconds is not None and \
                time.perf_counter() - started >= train.max_wall_seconds:
            return "max_wall_seconds"
        return None

    def _batch_seed(self, batch: UpdateBatch) -> int:
        import hashlib
        ident = ":".join([str(self.cfg.train.seed), str(batch.epoch),
                          str(batch.update_in_epoch), *batch.contrast_ids])
        return int.from_bytes(hashlib.sha256(ident.encode()).digest()[:4], "little")

    def _log(self, record: dict) -> None:
        if not self.dist.is_main:
            return
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _checkpoint(self) -> None:
        if not self.dist.is_main:
            return
        tmp = self.out / "checkpoint.pt.tmp"
        torch.save({
            "transform": self.transform.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "sampler": self.sampler.state_dict(),
            "completed": self.completed,
            "attempts": self.attempts,
            "best_metric": self.best_metric,
            "evals_without_improvement": self.evals_without_improvement,
            "config": self.cfg.to_dict(),
        }, tmp)
        tmp.replace(self.out / "checkpoint.pt")

    def _maybe_resume(self) -> None:
        path = self.out / "checkpoint.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.transform.load_state_dict(state["transform"])
        self.transform.to(self.backend.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.sampler.load_state_dict(state["sampler"])
        self.completed = state["completed"]
        self.attempts = state.get("attempts", self.completed)
        self.best_metric = state.get("best_metric", -float("inf"))
        self.evals_without_improvement = state.get("evals_without_improvement", 0)

    def _export(self, path: Path) -> None:
        save_transform(path, self.transform, provenance={
            "config": self.cfg.to_dict(),
            "config_fingerprint": self.cfg.fingerprint(),
            "completed_updates": self.completed,
        })


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")
