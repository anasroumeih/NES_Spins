from __future__ import annotations

import jax
import jax.numpy as jnp

from .lattice import num_sites
from .nes import batch_logabsdet


def _resolve_n_sites(shape: tuple[int, ...], n_sites: int | None = None) -> int:
    return int(num_sites(shape) if n_sites is None else n_sites)


def init_bundles(
    key,
    n_chains: int,
    k: int,
    shape: tuple[int, ...],
    move_type: str = "single_flip",
    n_sites: int | None = None,
):
    """Initial NES bundles with shape (n_chains, k, N).

    Replica configurations are random. For generic independently initialized
    state heads this gives a nonzero determinant with high probability.
    """
    N = _resolve_n_sites(shape, n_sites)

    if move_type == "single_flip":
        return (
            2
            * jax.random.bernoulli(
                key,
                0.5,
                (n_chains, k, N),
            ).astype(jnp.int8)
            - 1
        )

    if move_type == "pair_flip":
        if N % 2 != 0:
            raise ValueError("pair_flip/Sz=0 requires even N")

        base = jnp.concatenate(
            [
                jnp.ones(N // 2, dtype=jnp.int8),
                -jnp.ones(N // 2, dtype=jnp.int8),
            ]
        )

        keys = jax.random.split(key, n_chains * k)

        configs = jax.vmap(
            lambda kk: jax.random.permutation(kk, base)
        )(keys)

        return configs.reshape((n_chains, k, N))

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
):
    """Draw initial bundles until every chain has finite log|det A|.

    This does not alter the determinant. It only redraws invalid initial
    walkers, which have zero probability under the exact NES distribution.
    """
    key, key_init = jax.random.split(key)

    bundles = init_bundles(
        key_init,
        n_chains,
        k,
        shape,
        move_type,
        n_sites=n_sites,
    )

    for _ in range(max_retries):
        logabs = batch_logabsdet(apply_fun, params, bundles)
        invalid = ~jnp.isfinite(logabs)

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
        )

        bundles = jnp.where(
            invalid[:, None, None],
            replacement,
            bundles,
        )

    raise RuntimeError(
        "Could not initialize valid NES bundles. "
        "The model outputs may be linearly dependent at initialization."
    )


def init_configs(
    key,
    n_chains: int,
    shape: tuple[int, ...],
    move_type: str = "single_flip",
    n_sites: int | None = None,
):
    """Initial single-config chains, used for span-matrix evaluation."""
    N = _resolve_n_sites(shape, n_sites)

    if move_type == "single_flip":
        return (
            2
            * jax.random.bernoulli(
                key,
                0.5,
                (n_chains, N),
            ).astype(jnp.int8)
            - 1
        )

    if move_type == "pair_flip":
        if N % 2 != 0:
            raise ValueError("pair_flip/Sz=0 requires even N")

        base = jnp.concatenate(
            [
                jnp.ones(N // 2, dtype=jnp.int8),
                -jnp.ones(N // 2, dtype=jnp.int8),
            ]
        )

        keys = jax.random.split(key, n_chains)

        return jax.vmap(
            lambda kk: jax.random.permutation(kk, base)
        )(keys)

    raise ValueError(move_type)


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
):
    N = _resolve_n_sites(shape, n_sites)

    def sample(params, key, bundles):
        def step_with_params(carry, key):
            bundles, logabs = carry
            C = bundles.shape[0]

            key_rep, key_site1, key_site2, key_u = jax.random.split(key, 4)

            reps = jax.random.randint(key_rep, (C,), 0, k)
            site1 = jax.random.randint(key_site1, (C,), 0, N)
            rows = jnp.arange(C)

            if move_type == "single_flip":
                proposal = bundles.at[rows, reps, site1].multiply(-1)

            elif move_type == "pair_flip":
                site2 = jax.random.randint(key_site2, (C,), 0, N)

                s1 = bundles[rows, reps, site1]
                s2 = bundles[rows, reps, site2]

                active = (site1 != site2) & (s1 != s2)

                flipped = (
                    bundles.at[rows, reps, site1]
                    .multiply(-1)
                    .at[rows, reps, site2]
                    .multiply(-1)
                )

                proposal = jnp.where(
                    active[:, None, None],
                    flipped,
                    bundles,
                )

            else:
                raise ValueError(move_type)

            new_logabs = batch_logabsdet(
                apply_fun,
                params,
                proposal,
            )

            current_valid = jnp.isfinite(logabs)
            proposal_valid = jnp.isfinite(new_logabs)

            # Exact NES rejection rule:
            # singular proposal => probability zero => reject.
            #
            # If a bad initial walker somehow exists, accept any valid
            # proposal so it can leave the zero-weight state.
            logu = jnp.log(
                jax.random.uniform(key_u, (C,)) + 1e-12
            )

            log_ratio = 2.0 * (new_logabs - logabs)

            normal_accept = (
                current_valid
                & proposal_valid
                & (logu < jnp.minimum(log_ratio, 0.0))
            )

            rescue_accept = (~current_valid) & proposal_valid

            accept = normal_accept | rescue_accept

            bundles = jnp.where(
                accept[:, None, None],
                proposal,
                bundles,
            )

            logabs = jnp.where(
                accept,
                new_logabs,
                logabs,
            )

            return (
                bundles,
                logabs,
            ), jnp.mean(accept.astype(jnp.float32))

        logabs0 = batch_logabsdet(
            apply_fun,
            params,
            bundles,
        )

        keys = jax.random.split(key, burn_in)

        (bundles, logabs), acc_burn = jax.lax.scan(
            step_with_params,
            (bundles, logabs0),
            keys,
        )

        def collect(carry, key):
            bundles, logabs = carry

            keys_inner = jax.random.split(key, sweep_steps)

            (bundles, logabs), acc = jax.lax.scan(
                step_with_params,
                (bundles, logabs),
                keys_inner,
            )

            return (
                bundles,
                logabs,
            ), (
                bundles,
                jnp.mean(acc),
            )

        keys_collect = jax.random.split(key, n_samples)

        (bundles, logabs), (samples, accs) = jax.lax.scan(
            collect,
            (bundles, logabs),
            keys_collect,
        )

        flat_samples = samples.reshape(
            (n_samples * n_chains, k, N)
        )

        stats = {
            "accept_rate": jnp.mean(accs),
            "burn_accept_rate": (
                jnp.mean(acc_burn)
                if burn_in > 0
                else jnp.array(0.0)
            ),
            "invalid_final_fraction": jnp.mean(
                (~jnp.isfinite(logabs)).astype(jnp.float32)
            ),
        }

        return flat_samples, bundles, stats

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
):
    """Sampler for q(sigma) ∝ sum_i psi_i(sigma)^2, used for span evaluation."""
    N = _resolve_n_sites(shape, n_sites)

    def q_log(params, configs):
        vals = apply_fun(params, configs)
        q = jnp.sum(vals * vals, axis=-1) + eps
        return jnp.log(q)

    def sample(params, key, configs):
        def step_with_params(carry, key):
            configs, logq = carry
            C = configs.shape[0]

            key_site1, key_site2, key_u = jax.random.split(key, 3)

            site1 = jax.random.randint(key_site1, (C,), 0, N)
            rows = jnp.arange(C)

            if move_type == "single_flip":
                proposal = configs.at[rows, site1].multiply(-1)

            elif move_type == "pair_flip":
                site2 = jax.random.randint(key_site2, (C,), 0, N)

                s1 = configs[rows, site1]
                s2 = configs[rows, site2]

                active = (site1 != site2) & (s1 != s2)

                flipped = (
                    configs.at[rows, site1]
                    .multiply(-1)
                    .at[rows, site2]
                    .multiply(-1)
                )

                proposal = jnp.where(active[:, None], flipped, configs)

            else:
                raise ValueError(move_type)

            new_logq = q_log(params, proposal)

            logu = jnp.log(
                jax.random.uniform(key_u, (C,)) + 1e-12
            )

            accept = logu < (new_logq - logq)

            configs = jnp.where(
                accept[:, None],
                proposal,
                configs,
            )

            logq = jnp.where(
                accept,
                new_logq,
                logq,
            )

            return (
                configs,
                logq,
            ), jnp.mean(accept.astype(jnp.float32))

        logq0 = q_log(params, configs)

        keys = jax.random.split(key, burn_in)

        (configs, logq), acc_burn = jax.lax.scan(
            step_with_params,
            (configs, logq0),
            keys,
        )

        def collect(carry, key):
            configs, logq = carry

            keys_inner = jax.random.split(key, sweep_steps)

            (configs, logq), acc = jax.lax.scan(
                step_with_params,
                (configs, logq),
                keys_inner,
            )

            return (
                configs,
                logq,
            ), (
                configs,
                jnp.mean(acc),
            )

        keys_collect = jax.random.split(key, n_samples)

        (configs, logq), (samples, accs) = jax.lax.scan(
            collect,
            (configs, logq),
            keys_collect,
        )

        flat_samples = samples.reshape(
            (n_samples * n_chains, N)
        )

        stats = {
            "accept_rate": jnp.mean(accs),
            "burn_accept_rate": (
                jnp.mean(acc_burn)
                if burn_in > 0
                else jnp.array(0.0)
            ),
        }

        return flat_samples, configs, stats

    return jax.jit(sample)