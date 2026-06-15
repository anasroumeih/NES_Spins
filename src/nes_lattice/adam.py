from __future__ import annotations

import jax
import jax.numpy as jnp


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def init_adam(params):
    return {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": jnp.array(0)}


def adam_step(params, grads, opt, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), opt["v"], grads)
    mhat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**t), m)
    vhat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**t), v)
    params = jax.tree_util.tree_map(lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


def clip_grads(grads, max_norm: float | None):
    if max_norm is None or max_norm <= 0:
        return grads, jnp.array(0.0)
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum([jnp.sum(x * x) for x in leaves]) + 1e-12)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda g: scale * g, grads), norm
