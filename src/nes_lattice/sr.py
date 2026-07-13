from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .nes import batch_logabsdet


@dataclass(frozen=True)
class SRInfo:
    """Small diagnostic container for stochastic reconfiguration."""

    residual_norm: jax.Array
    update_norm: jax.Array
    grad_norm: jax.Array


def _flat_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.vdot(x, x).real + 1e-30)


def _cg_solve_fixed(matvec, b: jnp.ndarray, *, maxiter: int, tol: float):
    """Small fixed-iteration conjugate-gradient solver.

    This avoids depending on scipy/optax and works inside jax.jit.  The loop is
    fixed-length, but the updates are masked once the residual is below tol.
    """
    x0 = jnp.zeros_like(b)
    r0 = b - matvec(x0)
    p0 = r0
    rs0 = jnp.vdot(r0, r0).real
    tol2 = jnp.asarray(tol * tol, dtype=rs0.dtype)

    def body(_, state):
        x, r, p, rs = state
        Ap = matvec(p)
        denom = jnp.vdot(p, Ap).real + 1e-30
        alpha = rs / denom
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.vdot(r_new, r_new).real
        beta = rs_new / (rs + 1e-30)
        p_new = r_new + beta * p
        active = rs > tol2
        x = jnp.where(active, x_new, x)
        r = jnp.where(active, r_new, r)
        p = jnp.where(active, p_new, p)
        rs = jnp.where(active, rs_new, rs)
        return x, r, p, rs

    x, r, _, rs = jax.lax.fori_loop(0, int(maxiter), body, (x0, r0, p0, rs0))
    return x, jnp.sqrt(rs + 1e-30)


def sr_precondition_gradient(
    apply_fun,
    params,
    bundles: jnp.ndarray,
    grad_tree,
    *,
    diag_shift: float = 1e-3,
    cg_iters: int = 50,
    cg_tol: float = 1e-6,
):
    """Apply stochastic reconfiguration / natural-gradient preconditioning.

    The NES sampling distribution is proportional to |det A|^2.  For a bundle
    X, define

        O_a(X) = d log|det A_theta(X)| / d theta_a.

    The SR metric used here is the covariance of O over the sampled bundles,

        S_ab = <(O_a-<O_a>)(O_b-<O_b>)> + diag_shift * delta_ab.

    Given the ordinary VMC gradient g, this returns delta solving

        S delta = g.

    The caller should update params as params <- params - lr * delta.
    This is penalty-free: it only changes the parameter-space metric.
    """
    grad_flat, unravel = ravel_pytree(grad_tree)

    def logabs_batch(p):
        return batch_logabsdet(apply_fun, p, bundles)

    def metric_matvec(v_flat: jnp.ndarray) -> jnp.ndarray:
        v_tree = unravel(v_flat)
        _, jvp_vals = jax.jvp(logabs_batch, (params,), (v_tree,))
        centered_jvp = jvp_vals - jnp.mean(jvp_vals)

        def scalar_fn(p):
            vals = logabs_batch(p)
            vals = vals - jnp.mean(vals)
            return jnp.mean(vals * jax.lax.stop_gradient(centered_jvp))

        mtree = jax.grad(scalar_fn)(params)
        mflat, _ = ravel_pytree(mtree)
        return mflat + jnp.asarray(diag_shift, dtype=mflat.dtype) * v_flat

    update_flat, residual = _cg_solve_fixed(
        metric_matvec,
        grad_flat,
        maxiter=int(cg_iters),
        tol=float(cg_tol),
    )
    update_tree = unravel(update_flat)
    info = SRInfo(
        residual_norm=residual,
        update_norm=_flat_norm(update_flat),
        grad_norm=_flat_norm(grad_flat),
    )
    return update_tree, info
