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
from .sampler import init_bundles, make_bundle_sampler


@dataclass
class TrainConfig:
    # Lattice/system. 2D is the default; 1D is the exception, e.g. shape=(10,).
    shape: tuple[int, ...] = (4, 4)
    hamiltonian: Literal["tfim", "heisenberg", "toric_code", "toric"] = "tfim"
    k: int = 2
    J: float = 1.0
    g: float = 1.0
    Je: float | None = None      # toric-code star coupling; defaults to J
    Jm: float | None = None      # toric-code plaquette coupling; defaults to J
    pbc: bool = True
    magnetization: int | None = None

    # Model choices: "ffn", "rbm", "cnn".
    model: Literal["ffn", "rbm", "cnn"] = "ffn"
    hidden: tuple[int, ...] = (64, 64)
    rbm_hidden: int = 32
    channels: tuple[int, ...] = (16, 16)
    kernel_size: int = 3
    init_scale: float = 0.05
    dtype: str = "float32"

    # Stochastic NES-VMC.
    steps: int = 1000
    lr: float = 2e-3
    n_chains: int = 64
    n_samples: int = 8          # collected per chain per optimization step => n_chains*n_samples bundles
    sweep_steps: int | None = None
    burn_in: int | None = None
    det_jitter: float = 1e-8
    grad_clip: float | None = 10.0

    # Evaluation/logging.
    print_every: int = 100
    eval_exact_if_sites_leq: int = 16
    eval_samples: int = 32
    eval_chains: int = 128
    reference: Literal["auto", "netket", "ed", "own_ed", "toric", "analytic", "none"] = "auto"
    own_ed_max_sites: int = 14
    netket_max_states: int = 2_000_000
    jitter: float = 1e-8

    seed: int = 0

    def __post_init__(self):
        self.shape = normalize_shape(self.shape)
        self.hidden = tuple(self.hidden)
        self.channels = tuple(self.channels)
        n_move_sites = 2 * num_sites(self.shape) if str(self.hamiltonian).lower() in ("toric", "tc", "toric_code") else num_sites(self.shape)
        if self.sweep_steps is None:
            self.sweep_steps = max(1, n_move_sites)
        if self.burn_in is None:
            self.burn_in = 10 * max(1, n_move_sites)


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
    bundles = init_bundles(key_init, cfg.n_chains, cfg.k, cfg.shape, hspec.move_type, n_sites=hspec.N)

    sample_fn = make_bundle_sampler(
        apply_fun=apply_fun,
        shape=cfg.shape,
        k=cfg.k,
        move_type=hspec.move_type,
        n_chains=cfg.n_chains,
        n_samples=cfg.n_samples,
        sweep_steps=cfg.sweep_steps,
        burn_in=cfg.burn_in,
        det_jitter=cfg.det_jitter,
        n_sites=hspec.N,
    )

    @jax.jit
    def train_step(params, opt, samples):
        def loss_fn(p):
            loss, energy = vmc_surrogate_loss(apply_fun, p, samples, hspec, bonds, cfg.det_jitter)
            return loss, energy
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
            )
            loss_sum = float(np.sum(energies))
            if reference is not None:
                exact = [float(x) for x in reference]
                abs_errors = [float(abs(a - b)) for a, b in zip(energies, reference)]
                trace_error = float(abs(np.sum(energies[: len(reference)]) - np.sum(reference)))
            else:
                exact = None
                abs_errors = None
                trace_error = None
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
            last_grad_norm = float(grad_norm)

    return params, history


def save_history(history, path: str | Path, cfg: TrainConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(cfg), "history": history}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_history(path: str | Path):
    return json.loads(Path(path).read_text())
