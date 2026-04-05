from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_list(value: str):
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Empty list argument")
    return items


def main() -> None:
    p = argparse.ArgumentParser(description="Run pooling tradeoff grid over backbone/pooling combinations.")
    p.add_argument("--config", required=True)
    p.add_argument("--backbones", default="lightweight_cnn,acdnet")
    p.add_argument("--poolings", default="ssrp_t,adaptive_ssrp_t")
    p.add_argument("--device", default="auto")
    p.add_argument("--root", default=None)
    p.add_argument("--folds-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--input-representation", default=None, choices=["mel", "waveform"])
    args = p.parse_args()

    backbones = parse_list(args.backbones)
    poolings = parse_list(args.poolings)

    for backbone in backbones:
        for pooling in poolings:
            cmd = [
                sys.executable,
                "-m",
                "experiments.esc50_pooling_tradeoff.train_eval_cv",
                "--config",
                str(Path(args.config)),
                "--backbone",
                backbone,
                "--pooling",
                pooling,
                "--device",
                args.device,
            ]
            if args.root:
                cmd += ["--root", args.root]
            if args.folds_dir:
                cmd += ["--folds-dir", args.folds_dir]
            if args.out_dir:
                cmd += ["--out-dir", args.out_dir]
            if args.max_folds is not None:
                cmd += ["--max-folds", str(args.max_folds)]
            if args.freeze_backbone:
                cmd += ["--freeze-backbone"]
            if args.input_representation:
                cmd += ["--input-representation", args.input_representation]

            print("\n" + "=" * 80)
            print(f"[RUN] backbone={backbone} pooling={pooling}")
            print(" ".join(cmd))
            print("=" * 80)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

