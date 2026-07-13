from pathlib import Path
import sys

PROJECT_ROOT = Path.home() / "Desktop" / "Master Thesis" / "NES_Spins"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nes_lattice.train import TrainConfig, train, save_history
from nes_lattice.plots import plot_history, print_final


def run_experiment(cfg: TrainConfig, filename: str):
    params, history = train(cfg)

    save_path = PROJECT_ROOT / "results" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_history(history, save_path, cfg)
    print_final(save_path)
    plot_history(save_path)

    return params, history


def main():
    cfg1 = TrainConfig(
        shape=(4, 4),
        hamiltonian="toric_code",
        k=5,
        model="toric_rbm",
        rbm_hidden=256,
        init_scale=0.02,
        steps=2000,
        lr=1e-4,
        grad_clip=1.0,
        n_chains=512,
        n_samples=64,
        sweep_steps=128,
        burn_in=1024,
        toric_loop_prob=0.0,
        toric_single_flip_prob=0.0,
        toric_cover_sectors=True,
    )

    run_experiment(
        cfg1,
        "sampled_nes_toric_4x4_k5_rbm_notoricloops.json",
    )

    cfg2 = TrainConfig(
        shape=(4, 4),
        hamiltonian="toric_code",
        k=4,
        model="resffn",
        hidden=(512, 512, 512),
        init_scale=0.005,
        Je=1.0,
        Jm=1.0,
        steps=20000,
        lr=2e-5,
        grad_clip=0.5,
        n_chains=1024,
        n_samples=64,
        sweep_steps=128,
        burn_in=1024,
        toric_loop_prob=0.10,
        toric_single_flip_prob=0.0,
        toric_cover_sectors=True,
        print_every=500,
        eval_chains=1024,
        eval_samples=128,
        reference="auto",
        seed=4,
    )

    run_experiment(
        cfg2,
        "sampled_nes_toric_4x4_k4_resffn_toricmoves.json",
    )

    cfg3 = TrainConfig(
        shape=(4, 4),
        hamiltonian="toric_code",
        k=5,
        model="toric_rbm",
        rbm_hidden=256,
        init_scale=0.02,
        steps=1000,
        lr=1e-4,
        grad_clip=1.0,
        n_chains=512,
        n_samples=16,
        sweep_steps=128,
        burn_in=32,
        toric_loop_prob=0.15,
        toric_single_flip_prob=0.0,
        toric_cover_sectors=True,
        reference="ed",
        seed=4,
    )

    run_experiment(
        cfg3,
        "sampled_nes_toric_4x4_k5_rbm_toricmoves.json",
    )


if __name__ == "__main__":
    main()