"""CLI: build-data -> extract -> calibrate -> train -> evaluate.

    ste-optimized build-data --source /path/to/ESD --config configs/angry.yaml
    ste-optimized extract    --config configs/angry.yaml --split train
    ste-optimized calibrate  --config configs/angry.yaml
    ste-optimized train      --config configs/angry.yaml [--seed N] [--output DIR]
                             [--max-updates N] [--distributed ddp_rows]
    ste-optimized evaluate   --config configs/angry.yaml --transform PATH

Multi-GPU: see distributed.py (seed_parallel recommended; ddp_rows optional
under torchrun).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ste-optimized")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-data", help="pair ESD audio + choose bases + manifest")
    _add_common(p)
    p.add_argument("--source", required=True,
                   help="ESD root dir or generic CSV (path,speaker,emotion,text,index)")

    p = sub.add_parser("extract", help="batched mean-decode contrast extraction")
    _add_common(p)
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test"])
    p.add_argument("--batch-pairs", type=int, default=8)
    p.add_argument("--output", default=None)

    p = sub.add_parser("calibrate", help="machine micro-benchmarks (plan §5)")
    _add_common(p)
    p.add_argument("--output", default="calibration.json")

    p = sub.add_parser("train", help="batched expert-in-the-loop training")
    _add_common(p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--distributed", default=None,
                   choices=["none", "seed_parallel", "ddp_rows"])

    p = sub.add_parser("evaluate", help="full gated validation panel")
    _add_common(p)
    p.add_argument("--transform", required=True)
    p.add_argument("--output", default="panel_report.json")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "build-data":
        from .data import build_manifest
        manifest = build_manifest(args.source, cfg.data.emotion,
                                  cfg.data.dataset_dir,
                                  cfg.data.bases_per_speaker)
        print(f"pairs per split: {manifest['counts']}")
        print(f"bases per split: {manifest['base_counts']}")
        return 0

    if args.cmd == "extract":
        from .extraction import extract_contrasts
        out = args.output or (Path(cfg.data.dataset_dir)
                              / f"contrasts-{cfg.data.emotion}-{args.split}.pt")
        path = extract_contrasts(cfg, args.split, out,
                                 batch_pairs=args.batch_pairs)
        print(f"contrasts written: {path}")
        if args.split == "train" and not cfg.data.contrasts_path:
            print(f"set data.contrasts_path: {path} in your config")
        return 0

    if args.cmd == "calibrate":
        from .calibrate import run_calibration
        report = run_calibration(cfg, args.output)
        print(f"calibration written: {args.output}")
        for k in ("model_load_seconds", "generation", "tf_fwd_bwd"):
            print(f"  {k}: {report.get(k)}")
        return 0

    if args.cmd == "train":
        if args.seed is not None:
            cfg.train.seed = args.seed
        if args.output is not None:
            cfg.train.output_dir = args.output
        if args.max_updates is not None:
            cfg.train.max_updates = args.max_updates
        if args.distributed is not None:
            cfg.distributed.mode = args.distributed
        from .distributed import device_for_rank, init_distributed
        ctx = init_distributed(cfg.distributed.mode, cfg.distributed.backend)
        cfg.model.device = device_for_rank(ctx, cfg.model.device)
        from .evaluation import cadence_metric
        from .training import Trainer
        trainer = Trainer(cfg, ctx)
        trainer.train(cadence_eval=cadence_metric)
        return 0

    if args.cmd == "evaluate":
        from .evaluation import full_panel
        from .transform import load_transform
        transform, _prov = load_transform(args.transform)
        report = full_panel(cfg, transform, args.output)
        print(json.dumps(report["gates"], indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
