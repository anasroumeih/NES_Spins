from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nes_lattice.train import TrainConfig, save_history, train  # noqa: E402


CANDIDATES = [
    {
        "name": "toric_w128_s0005_lr1e-4_star_seed7",
        "model": "toric_resffn",
        "hidden": (128, 128),
        "init_scale": 0.005,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.0,
        "seed": 7,
    },
    {
        "name": "w128_s0005_lr1e-4_star_seed1",
        "hidden": (128, 128),
        "init_scale": 0.005,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.0,
        "seed": 1,
    },
    {
        "name": "w128_s001_lr1e-4_star_seed2",
        "hidden": (128, 128),
        "init_scale": 0.01,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.0,
        "seed": 2,
    },
    {
        "name": "w128_s002_lr1e-4_star_seed3",
        "hidden": (128, 128),
        "init_scale": 0.02,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.0,
        "seed": 3,
    },
    {
        "name": "w128_s0005_lr5e-4_star_seed4",
        "hidden": (128, 128),
        "init_scale": 0.005,
        "lr": 5e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.0,
        "seed": 4,
    },
    {
        "name": "w128_s0005_lr1e-4_loop010_seed5",
        "hidden": (128, 128),
        "init_scale": 0.005,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.10,
        "seed": 5,
    },
    {
        "name": "w256_s001_lr1e-4_star_seed6",
        "hidden": (256, 256),
        "init_scale": 0.01,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "toric_loop_prob": 0.0,
        "seed": 6,
    },
]


def make_config(candidate: dict, steps: int, n_chains: int, n_samples: int) -> TrainConfig:
    return TrainConfig(
        shape=(4, 4),
        hamiltonian="toric_code",
        k=4,
        model=candidate.get("model", "resffn"),
        hidden=candidate["hidden"],
        init_scale=candidate["init_scale"],
        Je=1.0,
        Jm=1.0,
        steps=steps,
        lr=candidate["lr"],
        grad_clip=candidate["grad_clip"],
        n_chains=n_chains,
        n_samples=n_samples,
        sweep_steps=32,
        burn_in=64,
        toric_loop_prob=candidate["toric_loop_prob"],
        toric_single_flip_prob=0.0,
        toric_cover_sectors=True,
        print_every=steps,
        eval_chains=max(128, n_chains),
        eval_samples=8,
        reference="auto",
        seed=candidate["seed"],
    )


def summarize(name: str, cfg: TrainConfig, history: list[dict], seconds: float) -> dict:
    final = history[-1]
    abs_errors = final.get("abs_errors") or []
    return {
        "name": name,
        "seconds": seconds,
        "steps": final["step"],
        "energies": final["energies"],
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "trace_error": final.get("trace_error"),
        "S_rank": final.get("eval", {}).get("S_rank"),
        "condition_number_S": final.get("condition_number_S"),
        "sampler_accept_rate": final.get("sampler_accept_rate"),
        "grad_norm": final.get("grad_norm"),
        "config": asdict(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "resffn_toric_tuning",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        candidate
        for candidate in CANDIDATES
        if args.only is None or candidate["name"] in set(args.only)
    ]

    summaries = []
    for candidate in selected:
        cfg = make_config(candidate, args.steps, args.n_chains, args.n_samples)
        print(f"\n=== {candidate['name']} ===")
        t0 = time.time()
        _, history = train(cfg)
        seconds = time.time() - t0
        history_path = args.out_dir / f"{candidate['name']}_steps{args.steps}.json"
        save_history(history, history_path, cfg)
        summary = summarize(candidate["name"], cfg, history, seconds)
        summary["history_path"] = str(history_path)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    summaries.sort(
        key=lambda row: (
            float("inf") if row["max_abs_error"] is None else row["max_abs_error"],
            float("inf") if row["trace_error"] is None else abs(row["trace_error"]),
        )
    )
    summary_path = args.out_dir / f"summary_steps{args.steps}.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
