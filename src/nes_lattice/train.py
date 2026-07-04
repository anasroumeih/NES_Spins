from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .adam import init_adam, adam_step, clip_grads
from .evaluation import evaluate_span
from .hamiltonians import make_hamiltonian_spec
from .lattice import normalize_shape, num_sites
from .models import ModelSpec, init_model, apply_model
from .nes import vmc_surrogate_loss
from .references import get_reference_energies
from .sampler import initialize_valid_bundles, make_bundle_sampler


@dataclass
class TrainConfig:
    shape: tuple[int, ...] = (4, 4)
    hamiltonian: Literal["tfim", "heisenberg", "toric_code", "toric"] = "tfim"
    k: int = 2
    J: float = 1.0
    g: float = 1.0
    Je: float | None = None
    Jm: float | None = None
    pbc: bool = True
    magnetization: int | None = None

    # Ansatz.
    model: Literal["ffn", "rbm", "cnn", "vit"] = "ffn"
    hidden: tuple[int, ...] = (64, 64)
    rbm_hidden: int = 32
    channels: tuple[int, ...] = (16, 16)
    kernel_size: int = 3
    vit_patch_size: int = 2
    vit_d_model: int = 64
    vit_num_layers: int = 2
    vit_num_heads: int = 4
    vit_mlp_ratio: int = 2
    vit_use_positional_embeddings: bool = True
    vit_log_amplitude_clip: float = 20.0
    init_scale: float = 0.05
    dtype: str = "float32"

    # Stochastic exact-NES determinant sampling; no determinant jitter.
    steps: int = 1000
    lr: float = 2e-3
    n_chains: int = 64
    n_samples: int = 8
    sweep_steps: int | None = None
    burn_in: int | None = None
    grad_clip: float | None = 10.0

    # Toric-code ground-manifold sampler.  With single_flip_prob=0, the
    # star/loop kernel remains in B_p=+1 and is appropriate for k<=4 ground states.
    toric_loop_prob: float = 0.10
    toric_single_flip_prob: float = 0.0
    toric_cover_sectors: bool = True

    # Evaluation/logging. This is still the optional Ritz-span diagnostic.
    print_every: int = 100
    eval_exact_if_sites_leq: int = 16
    eval_samples: int = 32
    eval_chains: int = 128
    reference: Literal["auto", "netket", "ed", "own_ed", "toric", "analytic", "none"] = "auto"
    own_ed_max_sites: int = 14
    netket_max_states: int = 2_000_000
    jitter: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        self.shape = normalize_shape(self.shape)
        self.hidden = tuple(self.hidden)
        self.channels = tuple(self.channels)
        toric = str(self.hamiltonian).lower() in ("toric", "tc", "toric_code")
        n_move_sites = 2 * num_sites(self.shape) if toric else num_sites(self.shape)
        if self.sweep_steps is None:
            self.sweep_steps = max(1, n_move_sites)
        if self.burn_in is None:
            self.burn_in = 10 * max(1, n_move_sites)
        if self.model == "vit" and len(self.shape) != 2:
            raise ValueError("model='vit' requires a 2D shape, for example shape=(4, 4).")
        if not (0.0 <= self.toric_loop_prob <= 1.0):
            raise ValueError("toric_loop_prob must lie in [0, 1].")
        if not (0.0 <= self.toric_single_flip_prob <= 1.0):
            raise ValueError("toric_single_flip_prob must lie in [0, 1].")
        if self.toric_loop_prob + self.toric_single_flip_prob > 1.0:
            raise ValueError(
                "toric_loop_prob + toric_single_flip_prob must be <= 1; "
                "the remaining probability is used for star moves."
            )


def make_apply_fun(mspec: ModelSpec):
    return lambda params, spins: apply_model(params, spins, mspec)


def train(cfg: TrainConfig):
    cfg.__post_init__()
    hspec = make_hamiltonian_spec(
        name=cfg.hamiltonian,
        shape=cfg.shape,
        J=cfg.J,
        g=cfg.g,
        pbc=cfg.pbc,
        magnetization=cfg.magnetization,
        Je=cfg.Je,
        Jm=cfg.Jm,
    )
    bonds = jnp.asarray(hspec.bonds_np)
    mspec = ModelSpec(
        model=cfg.model,
        shape=cfg.shape,
        k=cfg.k,
        hidden=cfg.hidden,
        rbm_hidden=cfg.rbm_hidden,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        vit_patch_size=cfg.vit_patch_size,
        vit_d_model=cfg.vit_d_model,
        vit_num_layers=cfg.vit_num_layers,
        vit_num_heads=cfg.vit_num_heads,
        vit_mlp_ratio=cfg.vit_mlp_ratio,
        vit_use_positional_embeddings=cfg.vit_use_positional_embeddings,
        vit_log_amplitude_clip=cfg.vit_log_amplitude_clip,
        scale=cfg.init_scale,
        n_sites=hspec.N,
        input_channels=hspec.model_input_channels,
        dtype=cfg.dtype,
    )
    apply_fun = make_apply_fun(mspec)

    key = jax.random.PRNGKey(cfg.seed)
    key, key_params, key_init = jax.random.split(key, 3)
    params = init_model(key_params, mspec)
    opt = init_adam(params)
    bundles, _ = initialize_valid_bundles(
        apply_fun=apply_fun,
        params=params,
        key=key_init,
        n_chains=cfg.n_chains,
        k=cfg.k,
        shape=cfg.shape,
        move_type=hspec.move_type,
        n_sites=hspec.N,
        toric_cover_sectors=cfg.toric_cover_sectors,
    )

    sample_fn = make_bundle_sampler(
        apply_fun=apply_fun,
        shape=cfg.shape,
        k=cfg.k,
        move_type=hspec.move_type,
        n_chains=cfg.n_chains,
        n_samples=cfg.n_samples,
        sweep_steps=cfg.sweep_steps,
        burn_in=cfg.burn_in,
        n_sites=hspec.N,
        toric_loop_prob=cfg.toric_loop_prob,
        toric_single_flip_prob=cfg.toric_single_flip_prob,
    )

    @jax.jit
    def train_step(params, opt, samples):
        def loss_fn(p):
            return vmc_surrogate_loss(apply_fun, p, samples, hspec, bonds)
        (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads, grad_norm = clip_grads(grads, cfg.grad_clip)
        params, opt = adam_step(params, grads, opt, cfg.lr)
        return params, opt, loss, energy, grad_norm

    reference, reference_source = get_reference_energies(
        hspec,
        cfg.k,
        prefer=cfg.reference,
        own_ed_max_sites=cfg.own_ed_max_sites,
        netket_max_states=cfg.netket_max_states,
    )
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)

    history = []
    last_train_energy = np.nan
    last_accept = np.nan
    last_grad_norm = np.nan
    last_invalid_fraction = np.nan
    last_star_accept = np.nan
    last_loop_accept = np.nan
    last_single_accept = np.nan
    last_star_fraction = np.nan
    last_loop_fraction = np.nan
    last_single_fraction = np.nan

    for step in range(cfg.steps + 1):
        if step % cfg.print_every == 0 or step == cfg.steps:
            key, key_eval = jax.random.split(key)
            energies, cond_S, eval_stats = evaluate_span(
                apply_fun,
                params,
                hspec,
                bonds,
                key_eval,
                exact_if_sites_leq=cfg.eval_exact_if_sites_leq,
                eval_samples=cfg.eval_samples,
                eval_chains=cfg.eval_chains,
                jitter=cfg.jitter,
                toric_loop_prob=cfg.toric_loop_prob,
                toric_single_flip_prob=cfg.toric_single_flip_prob,
            )
            loss_sum = float(np.sum(energies))
            if reference is not None:
                exact = [float(x) for x in reference]
                abs_errors = [float(abs(a - b)) for a, b in zip(energies, reference)]
                trace_error = float(abs(np.sum(energies[:len(reference)]) - np.sum(reference)))
            else:
                exact = abs_errors = trace_error = None
            rec = {
                "step": int(step),
                "loss_sum": loss_sum,
                "train_energy_estimator": float(last_train_energy),
                "energies": [float(x) for x in energies],
                "reference": exact,
                "reference_source": reference_source,
                "abs_errors": abs_errors,
                "trace_error": trace_error,
                "condition_number_S": float(cond_S),
                "sampler_accept_rate": float(last_accept),
                "sampler_star_accept_rate": float(last_star_accept),
                "sampler_loop_accept_rate": float(last_loop_accept),
                "sampler_single_flip_accept_rate": float(last_single_accept),
                "sampler_star_move_fraction": float(last_star_fraction),
                "sampler_loop_move_fraction": float(last_loop_fraction),
                "sampler_single_flip_move_fraction": float(last_single_fraction),
                "invalid_bundle_fraction": float(last_invalid_fraction),
                "grad_norm": float(last_grad_norm),
                "eval": eval_stats,
            }
            history.append(rec)
            print(rec)

        if step < cfg.steps:
            key, key_sample = jax.random.split(key)
            samples, bundles, stats = sample_fn(params, key_sample, bundles)
            params, opt, loss, energy, grad_norm = train_step(params, opt, samples)
            last_train_energy = float(energy)
            last_accept = float(stats["accept_rate"])
            last_star_accept = float(stats.get("star_accept_rate", np.nan))
            last_loop_accept = float(stats.get("loop_accept_rate", np.nan))
            last_single_accept = float(stats.get("single_flip_accept_rate", np.nan))
            last_star_fraction = float(stats.get("star_move_fraction", np.nan))
            last_loop_fraction = float(stats.get("loop_move_fraction", np.nan))
            last_single_fraction = float(stats.get("single_flip_move_fraction", np.nan))
            last_invalid_fraction = float(stats["invalid_final_fraction"])
            last_grad_norm = float(grad_norm)

    return params, history


def save_history(history, path: str | Path, cfg: TrainConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=2))
    return path


def load_history(path: str | Path):
    return json.loads(Path(path).read_text())
