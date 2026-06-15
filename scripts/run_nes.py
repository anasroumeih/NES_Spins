from pathlib import Path

from nes_lattice.train import TrainConfig, train, save_history


cfg = TrainConfig(
    shape=(4, 4),
    hamiltonian="tfim",
    k=2,
    g=1.0,
    model="ffn",
    hidden=(64, 64),
    steps=500,
    lr=2e-3,
    n_chains=64,
    n_samples=8,
    print_every=100,
    seed=0,
)

params, history = train(cfg)
out = Path("results") / "sampled_nes_2d_tfim_4x4_k2_ffn.json"
save_history(history, out, cfg)
print("saved to", out.resolve())
