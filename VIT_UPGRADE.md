# ViT upgrade

## New files

```text
src/nes_lattice/vit.py
notebooks/06_vit_nes.ipynb
scripts/smoke_test_vit.py
scripts/run_vit_tfim.py
```

## Modified files

```text
src/nes_lattice/models.py
src/nes_lattice/train.py
src/nes_lattice/nes.py
src/nes_lattice/sampler.py
src/nes_lattice/plots.py
pyproject.toml
requirements.txt
README.md
```

`nes.py` and `sampler.py` in this archive include the prior exact-`slogdet` / singular-proposal-rejection update.  They contain no determinant jitter.

## Install / check

```bash
cd ~/path/to/nes_lattice_project
source ~/myenv/bin/activate
pip install -e .
python scripts/smoke_test_vit.py
```

Expected output includes finite arrays of shape `(3, 2)` for the TFIM case and `(3, 4)` for the toric case.

## Notebook

Open `notebooks/06_vit_nes.ipynb`.  It contains:

1. ViT initialization and forward-pass test;
2. a `2x2` TFIM exact-span smoke test;
3. a `4x4` TFIM run that saves to `results/sampled_nes_tfim_4x4_k2_vit.json`;
4. the standard `print_final`, `plot_history`, and `plot_diagnostics` cells.

## ViT constraints

- `model="vit"` only accepts a two-dimensional `shape`.
- `vit_patch_size` must divide both dimensions; use `1` for `3x3`.
- TFIM uses one input channel. Toric code uses two edge-variable channels automatically.
- The current ViT produces real positive amplitudes. Use it for the stoquastic TFIM and toric code, not yet for a sign/phase-sensitive Heisenberg study.
