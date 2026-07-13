from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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
from .sr import sr_precondition_gradient
from .variance import variance_from_bundles


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

    # Ansatz.  Keep this as str so local experimental models such as
    # toric_rbm / toric_rbm_biasinit do not require editing this file again.
    model: str = "ffn"
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

    # Optional toric-RBM-biasinit fields.  These are only passed to ModelSpec if
    # your local models.py contains matching fields.
    toric_visible_bias_scale: float = 0.12
    toric_hidden_bias_scale: float = 0.10
    toric_weight_scale: float = 0.03

    dtype: str = "float32"

    # Stochastic exact-NES determinant sampling; no determinant jitter.
    steps: int = 1000
    lr: float = 2e-3
    n_chains: int = 64
    n_samples: int = 8
    sweep_steps: int | None = None
    burn_in: int | None = None
    grad_clip: float | None = 10.0

    # Optimizer.
    optimizer: Literal["adam", "sr"] = "adam"
    sr_diag_shift: float = 1e-3
    sr_cg_iters: int = 50
    sr_cg_tol: float = 1e-6

    # Learning-rate routine.
    lr_schedule: Literal["constant", "cosine", "linear", "inverse_sqrt", "step_decay"] = "constant"
    lr_final_factor: float = 0.1
    lr_decay_factor: float = 0.5
    lr_decay_every: int = 1000
    lr_warmup_steps: int = 0

    # Toric-code ground-manifold sampler.  With single_flip_prob=0, the
    # star/loop kernel remains in B_p=+1 and is appropriate for k<=4 ground states.
    toric_loop_prob: float = 0.10
    toric_single_flip_prob: float = 0.0
    toric_cover_sectors: bool = True

    # Evaluation/logging.  This is still the optional Ritz-span diagnostic.
    print_every: int = 100
    eval_exact_if_sites_leq: int = 16
    eval_samples: int = 32
    eval_chains: int = 128
    reference: Literal["auto", "netket", "ed", "own_ed", "toric", "analytic", "none"] = "auto"
    own_ed_max_sites: int = 14
    netket_max_states: int = 2_000_000
    jitter: float = 1e-6

    # NES S12 variance logging.  This computes variance from the local energy
    # matrix, not from the Ritz energies.
    log_variance: bool = True
    variance_every: int | None = None

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
        if self.variance_every is None:
            self.variance_every = self.print_every

        if self.model == "vit" and len(self.shape) != 2:
            raise ValueError("model='vit' requires a 2D shape, for example shape=(4, 4).")
        if self.optimizer not in ("adam", "sr"):
            raise ValueError("optimizer must be 'adam' or 'sr'.")
        if not (0.0 <= self.toric_loop_prob <= 1.0):
            raise ValueError("toric_loop_prob must lie in [0, 1].")
        if not (0.0 <= self.toric_single_flip_prob <= 1.0):
            raise ValueError("toric_single_flip_prob must lie in [0, 1].")
        if self.toric_loop_prob + self.toric_single_flip_prob > 1.0:
            raise ValueError(
                "toric_loop_prob + toric_single_flip_prob must be <= 1; "
                "the remaining probability is used for star moves."
            )
        if self.lr_decay_every <= 0:
            raise ValueError("lr_decay_every must be positive.")
        if self.sr_diag_shift <= 0.0:
            raise ValueError("sr_diag_shift must be positive.")


def make_apply_fun(mspec: ModelSpec):
    return lambda params, spins: apply_model(params, spins, mspec)


def _model_spec_from_config(cfg: TrainConfig, hspec) -> ModelSpec:
    kwargs = dict(
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
        toric_visible_bias_scale=cfg.toric_visible_bias_scale,
        toric_hidden_bias_scale=cfg.toric_hidden_bias_scale,
        toric_weight_scale=cfg.toric_weight_scale,
    )
    allowed = {f.name for f in fields(ModelSpec)}
    return ModelSpec(**{k: v for k, v in kwargs.items() if k in allowed})


def learning_rate_at_step(cfg: TrainConfig, step: int) -> float:
    """Learning-rate routine used by both Adam and SR."""
    if cfg.steps <= 0:
        progress = 1.0
    else:
        progress = min(max(float(step) / float(cfg.steps), 0.0), 1.0)

    if cfg.lr_schedule == "constant":
        lr = cfg.lr
    elif cfg.lr_schedule == "cosine":
        lr_final = cfg.lr * cfg.lr_final_factor
        lr = lr_final + 0.5 * (cfg.lr - lr_final) * (1.0 + np.cos(np.pi * progress))
    elif cfg.lr_schedule == "linear":
        lr_final = cfg.lr * cfg.lr_final_factor
        lr = cfg.lr + progress * (lr_final - cfg.lr)
    elif cfg.lr_schedule == "inverse_sqrt":
        lr = cfg.lr / np.sqrt(1.0 + step / float(cfg.lr_decay_every))
    elif cfg.lr_schedule == "step_decay":
        lr = cfg.lr * (cfg.lr_decay_factor ** (step // cfg.lr_decay_every))
    else:
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule}")

    if cfg.lr_warmup_steps and step < cfg.lr_warmup_steps:
        lr *= float(step + 1) / float(cfg.lr_warmup_steps)
    return float(lr)


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

    mspec = _model_spec_from_config(cfg, hspec)
    apply_fun = make_apply_fun(mspec)

    key = jax.random.PRNGKey(cfg.seed)
    key, key_params, key_init = jax.random.split(key, 3)
    params = init_model(key_params, mspec)
    opt = init_adam(params) if cfg.optimizer == "adam" else None

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
    def train_step_adam(params, opt, samples, lr_t):
        def loss_fn(p):
            return vmc_surrogate_loss(apply_fun, p, samples, hspec, bonds)

        (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads, grad_norm = clip_grads(grads, cfg.grad_clip)
        params, opt = adam_step(params, grads, opt, lr_t)
        return params, opt, loss, energy, grad_norm

    @jax.jit
    def train_step_sr(params, samples, lr_t):
        def loss_fn(p):
            return vmc_surrogate_loss(apply_fun, p, samples, hspec, bonds)

        (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        update, sr_info = sr_precondition_gradient(
            apply_fun,
            params,
            samples,
            grads,
            diag_shift=cfg.sr_diag_shift,
            cg_iters=cfg.sr_cg_iters,
            cg_tol=cfg.sr_cg_tol,
        )
        update, _ = clip_grads(update, cfg.grad_clip)
        params = jax.tree_util.tree_map(lambda p, u: p - lr_t * u, params, update)
        return params, loss, energy, sr_info.grad_norm, sr_info.update_norm, sr_info.residual_norm

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
    last_lr = np.nan
    last_sr_update_norm = np.nan
    last_sr_residual_norm = np.nan

    for step in range(cfg.steps + 1):
        do_print = (step % cfg.print_every == 0) or (step == cfg.steps)
        do_variance = bool(cfg.log_variance) and (
            (step % int(cfg.variance_every) == 0) or (step == cfg.steps)
        )

        if do_print:
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

            variance_stats = None
            if do_variance:
                key, key_var = jax.random.split(key)
                var_samples, _, var_sampler_stats = sample_fn(params, key_var, bundles)
                variance_stats = variance_from_bundles(
                    apply_fun,
                    params,
                    var_samples,
                    hspec,
                    bonds,
                )
                variance_stats["variance_sampler_accept_rate"] = float(var_sampler_stats["accept_rate"])

            loss_sum = float(np.sum(energies))
            if reference is not None:
                exact = [float(x) for x in reference]
                abs_errors = [float(abs(a - b)) for a, b in zip(energies, reference)]
                trace_error = float(abs(np.sum(energies[: len(reference)]) - np.sum(reference)))
            else:
                exact = abs_errors = trace_error = None

            rec = {
                "step": int(step),
                "optimizer": cfg.optimizer,
                "lr": float(last_lr),
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
                "sr_update_norm": float(last_sr_update_norm),
                "sr_residual_norm": float(last_sr_residual_norm),
                "eval": eval_stats,
            }
            if variance_stats is not None:
                rec.update(
                    {
                        "energy_matrix_eigvals": variance_stats["energy_matrix_eigvals"],
                        "state_energy_variances": variance_stats["state_energy_variances"],
                        "state_energy_std_errors": variance_stats["state_energy_std_errors"],
                        "n_variance_samples": variance_stats["n_variance_samples"],
                        "variance_eval": variance_stats,
                    }
                )
            history.append(rec)
            print(rec)

        if step < cfg.steps:
            lr_t = learning_rate_at_step(cfg, step)
            key, key_sample = jax.random.split(key)
            samples, bundles, stats = sample_fn(params, key_sample, bundles)

            if cfg.optimizer == "adam":
                params, opt, loss, energy, grad_norm = train_step_adam(params, opt, samples, lr_t)
                last_sr_update_norm = np.nan
                last_sr_residual_norm = np.nan
            else:
                params, loss, energy, grad_norm, sr_update_norm, sr_residual_norm = train_step_sr(
                    params, samples, lr_t
                )
                last_sr_update_norm = float(sr_update_norm)
                last_sr_residual_norm = float(sr_residual_norm)

            last_lr = float(lr_t)
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
