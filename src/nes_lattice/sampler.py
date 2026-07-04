from __future__ import annotations

import jax
import jax.numpy as jnp

from .lattice import num_sites, toric_code_move_masks
from .nes import batch_logabsdet


def _resolve_n_sites(shape: tuple[int, ...], n_sites: int | None = None) -> int:
    return int(num_sites(shape) if n_sites is None else n_sites)


def _toric_masks(shape: tuple[int, ...]):
    """JAX move masks: stars and the two non-contractible winding loops."""
    star_masks_np, loop_masks_np = toric_code_move_masks(shape)
    return (
        jnp.asarray(star_masks_np, dtype=jnp.int8),
        jnp.asarray(loop_masks_np, dtype=jnp.int8),
    )


def _flip_by_mask(configs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Flip every edge selected by a binary mask, preserving int8 spins."""
    factor = (1 - 2 * mask).astype(configs.dtype)
    return configs * factor


def _toric_configs_from_star_orbit(
    key,
    n_configs: int,
    shape: tuple[int, ...],
    sector_bits: jnp.ndarray | None = None,
):
    """Sample flux-free toric configurations from star products and loops.

    Starting from all + spins, arbitrary star products generate the local
    gauge orbit.  The two winding-loop bits select one of the four torus
    sectors.  Every returned configuration has B_p=+1 for every plaquette.
    """
    star_masks, loop_masks = _toric_masks(shape)
    n_stars, n_edges = star_masks.shape

    key_star, key_sector = jax.random.split(key)
    star_bits = jax.random.bernoulli(
        key_star,
        0.5,
        shape=(n_configs, n_stars),
    ).astype(jnp.int32)

    parity = (star_bits @ star_masks.astype(jnp.int32)) % 2

    if sector_bits is None:
        sector_bits = jax.random.bernoulli(
            key_sector,
            0.5,
            shape=(n_configs, 2),
        ).astype(jnp.int32)
    else:
        sector_bits = jnp.asarray(sector_bits, dtype=jnp.int32)
        if sector_bits.shape != (n_configs, 2):
            raise ValueError(
                f"sector_bits must have shape {(n_configs, 2)}, got {sector_bits.shape}."
            )

    parity = (parity + sector_bits @ loop_masks.astype(jnp.int32)) % 2
    return (1 - 2 * parity).astype(jnp.int8)


def _init_toric_bundles(
    key,
    n_chains: int,
    k: int,
    shape: tuple[int, ...],
    cover_sectors: bool,
):
    """Flux-free toric bundles, optionally covering sectors within each bundle."""
    key_sector, key_configs = jax.random.split(key)

    if cover_sectors:
        # For k=4, every initial bundle contains all four Z2 x Z2 sectors.
        # For k<4 this distributes replicas across sectors; for k>4 it wraps.
        offsets = jax.random.randint(key_sector, (n_chains,), 0, 4)
        labels = (offsets[:, None] + jnp.arange(k)[None, :]) % 4
    else:
        labels = jax.random.randint(key_sector, (n_chains, k), 0, 4)

    sector_bits = jnp.stack(
        [labels & 1, (labels >> 1) & 1],
        axis=-1,
    ).reshape(n_chains * k, 2)

    configs = _toric_configs_from_star_orbit(
        key_configs,
        n_chains * k,
        shape,
        sector_bits=sector_bits,
    )
    return configs.reshape(n_chains, k, -1)


def init_bundles(
    key,
    n_chains: int,
    k: int,
    shape: tuple[int, ...],
    move_type: str = "single_flip",
    n_sites: int | None = None,
    toric_cover_sectors: bool = True,
):
    """Initialize NES bundles for the selected sampler.

    ``move_type='toric'`` initializes only the flux-free B_p=+1 manifold.
    That is intentional for the four toric-code ground states.  Do not use
    it unchanged when targeting flux/anyon excitations.
    """
    N = _resolve_n_sites(shape, n_sites)

    if move_type == "toric":
        return _init_toric_bundles(
            key,
            n_chains,
            k,
            shape,
            cover_sectors=toric_cover_sectors,
        )

    if move_type == "single_flip":
        return (
            2
            * jax.random.bernoulli(key, 0.5, (n_chains, k, N)).astype(jnp.int8)
            - 1
        )

    if move_type == "pair_flip":
        if N % 2:
            raise ValueError("pair_flip/Sz=0 requires even N")
        base = jnp.concatenate(
            [jnp.ones(N // 2, dtype=jnp.int8), -jnp.ones(N // 2, dtype=jnp.int8)]
        )
        keys = jax.random.split(key, n_chains * k)
        return jax.vmap(lambda kk: jax.random.permutation(kk, base))(keys).reshape(
            n_chains, k, N
        )

    raise ValueError(move_type)


def initialize_valid_bundles(
    apply_fun,
    params,
    key,
    n_chains: int,
    k: int,
    shape: tuple[int, ...],
    move_type: str,
    n_sites: int | None = None,
    max_retries: int = 50,
    toric_cover_sectors: bool = True,
):
    """Redraw only zero-probability initial bundles; never modify det(A)."""
    key, key_init = jax.random.split(key)
    bundles = init_bundles(
        key_init,
        n_chains,
        k,
        shape,
        move_type,
        n_sites=n_sites,
        toric_cover_sectors=toric_cover_sectors,
    )

    for _ in range(max_retries):
        invalid = ~jnp.isfinite(batch_logabsdet(apply_fun, params, bundles))
        if not bool(jnp.any(invalid)):
            return bundles, key

        key, key_replace = jax.random.split(key)
        replacement = init_bundles(
            key_replace,
            n_chains,
            k,
            shape,
            move_type,
            n_sites=n_sites,
            toric_cover_sectors=toric_cover_sectors,
        )
        bundles = jnp.where(invalid[:, None, None], replacement, bundles)

    raise RuntimeError("Could not draw finite-det NES bundles at initialization.")


def init_configs(
    key,
    n_chains: int,
    shape: tuple[int, ...],
    move_type: str = "single_flip",
    n_sites: int | None = None,
):
    """Initialize ordinary configuration chains used by span diagnostics."""
    N = _resolve_n_sites(shape, n_sites)

    if move_type == "toric":
        return _toric_configs_from_star_orbit(key, n_chains, shape)

    if move_type == "single_flip":
        return 2 * jax.random.bernoulli(key, 0.5, (n_chains, N)).astype(jnp.int8) - 1

    if move_type == "pair_flip":
        if N % 2:
            raise ValueError("pair_flip/Sz=0 requires even N")
        base = jnp.concatenate(
            [jnp.ones(N // 2, dtype=jnp.int8), -jnp.ones(N // 2, dtype=jnp.int8)]
        )
        keys = jax.random.split(key, n_chains)
        return jax.vmap(lambda kk: jax.random.permutation(kk, base))(keys)

    raise ValueError(move_type)


def _validate_toric_probs(loop_prob: float, single_flip_prob: float):
    if not (0.0 <= loop_prob <= 1.0):
        raise ValueError("toric_loop_prob must lie in [0, 1].")
    if not (0.0 <= single_flip_prob <= 1.0):
        raise ValueError("toric_single_flip_prob must lie in [0, 1].")
    if loop_prob + single_flip_prob > 1.0:
        raise ValueError(
            "toric_loop_prob + toric_single_flip_prob must be <= 1. "
            "The remaining probability is used for star moves."
        )


def _move_metrics(
    accept: jnp.ndarray,
    is_star: jnp.ndarray,
    is_loop: jnp.ndarray,
    is_single: jnp.ndarray,
) -> jnp.ndarray:
    """[accepted,total, accepted_star,n_star, accepted_loop,n_loop, ...]."""
    accept_f = accept.astype(jnp.float32)
    star_f = is_star.astype(jnp.float32)
    loop_f = is_loop.astype(jnp.float32)
    single_f = is_single.astype(jnp.float32)
    total = jnp.asarray(accept.shape[0], dtype=jnp.float32)
    return jnp.stack(
        [
            jnp.sum(accept_f),
            total,
            jnp.sum(accept_f * star_f),
            jnp.sum(star_f),
            jnp.sum(accept_f * loop_f),
            jnp.sum(loop_f),
            jnp.sum(accept_f * single_f),
            jnp.sum(single_f),
        ]
    )


def _safe_rate(num: jnp.ndarray, den: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(den > 0.0, num / den, jnp.asarray(0.0, dtype=jnp.float32))


def _metrics_to_stats(metrics: jnp.ndarray, prefix: str = "") -> dict[str, jnp.ndarray]:
    return {
        f"{prefix}accept_rate": _safe_rate(metrics[0], metrics[1]),
        f"{prefix}star_accept_rate": _safe_rate(metrics[2], metrics[3]),
        f"{prefix}loop_accept_rate": _safe_rate(metrics[4], metrics[5]),
        f"{prefix}single_flip_accept_rate": _safe_rate(metrics[6], metrics[7]),
        f"{prefix}star_move_fraction": _safe_rate(metrics[3], metrics[1]),
        f"{prefix}loop_move_fraction": _safe_rate(metrics[5], metrics[1]),
        f"{prefix}single_flip_move_fraction": _safe_rate(metrics[7], metrics[1]),
    }


def make_bundle_sampler(
    apply_fun,
    shape: tuple[int, ...],
    k: int,
    move_type: str,
    n_chains: int,
    n_samples: int,
    sweep_steps: int,
    burn_in: int,
    n_sites: int | None = None,
    toric_loop_prob: float = 0.10,
    toric_single_flip_prob: float = 0.0,
):
    """Create a Metropolis sampler for determinant-weighted NES bundles.

    With ``move_type='toric'``, each proposal chooses a replica and then uses
    a star flip with probability ``1-loop_prob-single_flip_prob``, a winding
    loop with probability ``loop_prob`` (equally between the two directions),
    or an optional single-edge flip.  The first two preserve B_p exactly.
    """
    N = _resolve_n_sites(shape, n_sites)
    if move_type == "toric":
        _validate_toric_probs(toric_loop_prob, toric_single_flip_prob)
        star_masks, loop_masks = _toric_masks(shape)
        n_stars = star_masks.shape[0]

    def sample(params, key, bundles):
        def step_with_params(carry, key):
            bundles, logabs = carry
            C = bundles.shape[0]
            rows = jnp.arange(C)

            if move_type == "toric":
                key_rep, key_kind, key_star, key_loop, key_site, key_u = jax.random.split(key, 6)
                reps = jax.random.randint(key_rep, (C,), 0, k)
                move_u = jax.random.uniform(key_kind, (C,))
                star_id = jax.random.randint(key_star, (C,), 0, n_stars)
                loop_id = jax.random.randint(key_loop, (C,), 0, 2)
                site = jax.random.randint(key_site, (C,), 0, N)

                selected = bundles[rows, reps]
                star_selected = _flip_by_mask(selected, star_masks[star_id])
                loop_selected = _flip_by_mask(selected, loop_masks[loop_id])
                single_selected = selected.at[rows, site].multiply(-1)

                is_single = move_u < toric_single_flip_prob
                is_loop = (
                    (move_u >= toric_single_flip_prob)
                    & (move_u < toric_single_flip_prob + toric_loop_prob)
                )
                is_star = ~(is_single | is_loop)

                proposed_selected = jnp.where(
                    is_single[:, None],
                    single_selected,
                    jnp.where(is_loop[:, None], loop_selected, star_selected),
                )
                proposal = bundles.at[rows, reps].set(proposed_selected)
                key_u = key_u

            else:
                key_rep, key_site1, key_site2, key_u = jax.random.split(key, 4)
                reps = jax.random.randint(key_rep, (C,), 0, k)
                site1 = jax.random.randint(key_site1, (C,), 0, N)

                if move_type == "single_flip":
                    proposal = bundles.at[rows, reps, site1].multiply(-1)
                elif move_type == "pair_flip":
                    site2 = jax.random.randint(key_site2, (C,), 0, N)
                    active = (site1 != site2) & (
                        bundles[rows, reps, site1] != bundles[rows, reps, site2]
                    )
                    flipped = (
                        bundles.at[rows, reps, site1]
                        .multiply(-1)
                        .at[rows, reps, site2]
                        .multiply(-1)
                    )
                    proposal = jnp.where(active[:, None, None], flipped, bundles)
                else:
                    raise ValueError(move_type)

                is_star = jnp.zeros((C,), dtype=bool)
                is_loop = jnp.zeros((C,), dtype=bool)
                is_single = jnp.ones((C,), dtype=bool)

            new_logabs = batch_logabsdet(apply_fun, params, proposal)
            current_valid = jnp.isfinite(logabs)
            proposal_valid = jnp.isfinite(new_logabs)

            logu = jnp.log(jax.random.uniform(key_u, (C,)) + 1e-12)
            log_ratio = 2.0 * (new_logabs - logabs)
            normal_accept = current_valid & proposal_valid & (logu < jnp.minimum(log_ratio, 0.0))
            rescue_accept = (~current_valid) & proposal_valid
            accept = normal_accept | rescue_accept

            bundles = jnp.where(accept[:, None, None], proposal, bundles)
            logabs = jnp.where(accept, new_logabs, logabs)

            return (bundles, logabs), _move_metrics(accept, is_star, is_loop, is_single)

        logabs0 = batch_logabsdet(apply_fun, params, bundles)
        burn_keys = jax.random.split(key, burn_in)
        (bundles, logabs), burn_metrics = jax.lax.scan(
            step_with_params,
            (bundles, logabs0),
            burn_keys,
        )

        def collect(carry, subkey):
            bundles, logabs = carry
            inner = jax.random.split(subkey, sweep_steps)
            (bundles, logabs), metrics = jax.lax.scan(
                step_with_params,
                (bundles, logabs),
                inner,
            )
            return (bundles, logabs), (bundles, jnp.sum(metrics, axis=0))

        sample_keys = jax.random.split(key, n_samples)
        (bundles, logabs), (samples, sample_metrics) = jax.lax.scan(
            collect,
            (bundles, logabs),
            sample_keys,
        )

        metrics = jnp.sum(sample_metrics, axis=0)
        burn_total = jnp.sum(burn_metrics, axis=0)
        stats = _metrics_to_stats(metrics)
        stats.update(_metrics_to_stats(burn_total, prefix="burn_"))
        stats["invalid_final_fraction"] = jnp.mean((~jnp.isfinite(logabs)).astype(jnp.float32))

        return samples.reshape(n_samples * n_chains, k, N), bundles, stats

    return jax.jit(sample)


def make_config_sampler(
    apply_fun,
    shape: tuple[int, ...],
    move_type: str,
    n_chains: int,
    n_samples: int,
    sweep_steps: int,
    burn_in: int,
    eps: float = 1e-12,
    n_sites: int | None = None,
    toric_loop_prob: float = 0.10,
    toric_single_flip_prob: float = 0.0,
):
    """Sampler for q(sigma) proportional to sum_i psi_i(sigma)^2.

    Used only by the optional Ritz-span diagnostic.  For toric ground-space
    runs, it uses the same flux-free star/loop kernel as training.
    """
    N = _resolve_n_sites(shape, n_sites)
    if move_type == "toric":
        _validate_toric_probs(toric_loop_prob, toric_single_flip_prob)
        star_masks, loop_masks = _toric_masks(shape)
        n_stars = star_masks.shape[0]

    def q_log(params, configs):
        vals = apply_fun(params, configs)
        return jnp.log(jnp.sum(vals * vals, axis=-1) + eps)

    def sample(params, key, configs):
        def step_with_params(carry, key):
            configs, logq = carry
            C = configs.shape[0]
            rows = jnp.arange(C)

            if move_type == "toric":
                key_kind, key_star, key_loop, key_site, key_u = jax.random.split(key, 5)
                move_u = jax.random.uniform(key_kind, (C,))
                star_id = jax.random.randint(key_star, (C,), 0, n_stars)
                loop_id = jax.random.randint(key_loop, (C,), 0, 2)
                site = jax.random.randint(key_site, (C,), 0, N)

                star_prop = _flip_by_mask(configs, star_masks[star_id])
                loop_prop = _flip_by_mask(configs, loop_masks[loop_id])
                single_prop = configs.at[rows, site].multiply(-1)

                is_single = move_u < toric_single_flip_prob
                is_loop = (
                    (move_u >= toric_single_flip_prob)
                    & (move_u < toric_single_flip_prob + toric_loop_prob)
                )
                is_star = ~(is_single | is_loop)
                proposal = jnp.where(
                    is_single[:, None],
                    single_prop,
                    jnp.where(is_loop[:, None], loop_prop, star_prop),
                )

            else:
                key_site1, key_site2, key_u = jax.random.split(key, 3)
                site1 = jax.random.randint(key_site1, (C,), 0, N)
                if move_type == "single_flip":
                    proposal = configs.at[rows, site1].multiply(-1)
                elif move_type == "pair_flip":
                    site2 = jax.random.randint(key_site2, (C,), 0, N)
                    active = (site1 != site2) & (configs[rows, site1] != configs[rows, site2])
                    flipped = configs.at[rows, site1].multiply(-1).at[rows, site2].multiply(-1)
                    proposal = jnp.where(active[:, None], flipped, configs)
                else:
                    raise ValueError(move_type)
                is_star = jnp.zeros((C,), dtype=bool)
                is_loop = jnp.zeros((C,), dtype=bool)
                is_single = jnp.ones((C,), dtype=bool)

            new_logq = q_log(params, proposal)
            logu = jnp.log(jax.random.uniform(key_u, (C,)) + 1e-12)
            accept = logu < jnp.minimum(new_logq - logq, 0.0)
            configs = jnp.where(accept[:, None], proposal, configs)
            logq = jnp.where(accept, new_logq, logq)
            return (configs, logq), _move_metrics(accept, is_star, is_loop, is_single)

        logq0 = q_log(params, configs)
        burn_keys = jax.random.split(key, burn_in)
        (configs, logq), burn_metrics = jax.lax.scan(
            step_with_params,
            (configs, logq0),
            burn_keys,
        )

        def collect(carry, subkey):
            configs, logq = carry
            inner = jax.random.split(subkey, sweep_steps)
            (configs, logq), metrics = jax.lax.scan(
                step_with_params,
                (configs, logq),
                inner,
            )
            return (configs, logq), (configs, jnp.sum(metrics, axis=0))

        sample_keys = jax.random.split(key, n_samples)
        (configs, logq), (samples, sample_metrics) = jax.lax.scan(
            collect,
            (configs, logq),
            sample_keys,
        )

        metrics = jnp.sum(sample_metrics, axis=0)
        burn_total = jnp.sum(burn_metrics, axis=0)
        stats = _metrics_to_stats(metrics)
        stats.update(_metrics_to_stats(burn_total, prefix="burn_"))

        return samples.reshape(n_samples * n_chains, N), configs, stats

    return jax.jit(sample)
