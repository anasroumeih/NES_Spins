from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

ModelType = Literal["ffn", "rbm", "cnn"]


@dataclass(frozen=True)
class ModelSpec:
    model: ModelType = "ffn"
    shape: tuple[int, ...] = (4, 4)
    k: int = 2
    hidden: tuple[int, ...] = (64, 64)      # FFN hidden layers
    rbm_hidden: int = 32                    # RBM hidden units per state
    channels: tuple[int, ...] = (16, 16)    # CNN channels
    kernel_size: int = 3
    scale: float = 0.05
    dtype: str = "float32"                  # keep model params/input dtype consistent

    @property
    def N(self) -> int:
        n = 1
        for x in self.shape:
            n *= x
        return n


def _dtype(spec: ModelSpec):
    return jnp.dtype(spec.dtype)


def _param_dtype(params):
    """Return dtype of the first floating parameter leaf."""
    leaves = jax.tree_util.tree_leaves(params)
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            return leaf.dtype
    return jnp.float32


def init_model(key: jax.Array, spec: ModelSpec):
    if spec.model == "ffn":
        return init_ffn(key, spec.N, spec.k, spec.hidden, spec.scale, _dtype(spec))
    if spec.model == "rbm":
        return init_rbm(key, spec.N, spec.k, spec.rbm_hidden, spec.scale, _dtype(spec))
    if spec.model == "cnn":
        return init_cnn(key, spec.shape, spec.k, spec.channels, spec.kernel_size, spec.scale, _dtype(spec))
    raise ValueError(spec.model)


def apply_model(params, spins: jnp.ndarray, spec: ModelSpec) -> jnp.ndarray:
    if spec.model == "ffn":
        return apply_ffn(params, spins)
    if spec.model == "rbm":
        return apply_rbm(params, spins)
    if spec.model == "cnn":
        return apply_cnn(params, spins, spec.shape)
    raise ValueError(spec.model)


# ---------- FFN ----------

def init_ffn(
    key,
    N: int,
    k: int,
    hidden=(64, 64),
    scale: float = 0.05,
    dtype=jnp.float32,
):
    sizes = [N, *hidden, k]
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []

    for n_in, n_out, kk in zip(sizes[:-1], sizes[1:], keys):
        W = scale * jax.random.normal(
            kk,
            (n_in, n_out),
            dtype=dtype,
        ) / jnp.sqrt(jnp.asarray(n_in, dtype=dtype))

        b = jnp.zeros((n_out,), dtype=dtype)
        layers.append({"W": W, "b": b})

    layers[-1]["b"] = 0.01 * jnp.arange(k, dtype=dtype)
    return {"layers": layers}


def apply_ffn(params, spins):
    dtype = _param_dtype(params)
    x = spins.astype(dtype)

    for layer in params["layers"][:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])

    return x @ params["layers"][-1]["W"] + params["layers"][-1]["b"]


# ---------- RBM ----------

def init_rbm(
    key,
    N: int,
    k: int,
    n_hidden: int = 32,
    scale: float = 0.03,
    dtype=jnp.float32,
):
    k1, k2, k3 = jax.random.split(key, 3)

    # Separate RBM for each NES state.
    a = scale * jax.random.normal(k1, (k, N), dtype=dtype)
    b = scale * jax.random.normal(k2, (k, n_hidden), dtype=dtype)
    W = scale * jax.random.normal(
        k3,
        (k, N, n_hidden),
        dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(N, dtype=dtype))

    # State-dependent tiny prefactors break row symmetry.
    pref = 0.01 * jnp.arange(k, dtype=dtype)

    return {"a": a, "b": b, "W": W, "pref": pref}


def apply_rbm(params, spins):
    dtype = _param_dtype(params)
    s = spins.astype(dtype)

    # logpsi[..., state]
    visible = jnp.einsum("...n,kn->...k", s, params["a"])
    theta = jnp.einsum("...n,knh->...kh", s, params["W"]) + params["b"]

    # log(2 cosh theta), written stably
    hidden = jnp.sum(jnp.logaddexp(theta, -theta), axis=-1)

    logpsi = params["pref"] + visible + hidden

    # Remove baseline log(2)*M to keep amplitudes moderate.
    logpsi = logpsi - jnp.log(jnp.asarray(2.0, dtype=dtype)) * params["b"].shape[-1]

    return jnp.exp(jnp.clip(logpsi, -30.0, 30.0))


# ---------- CNN ----------

def init_cnn(
    key,
    shape: tuple[int, ...],
    k: int,
    channels=(16, 16),
    kernel_size: int = 3,
    scale: float = 0.05,
    dtype=jnp.float32,
):
    if len(shape) == 1:
        spatial_shape = (1, shape[0])
    elif len(shape) == 2:
        spatial_shape = shape
    else:
        raise ValueError("CNN supports only 1D or 2D")

    keys = jax.random.split(key, len(channels) + 1)

    convs = []
    in_ch = 1

    for out_ch, kk in zip(channels, keys[:-1]):
        fan_in = kernel_size * kernel_size * in_ch

        W = scale * jax.random.normal(
            kk,
            (kernel_size, kernel_size, in_ch, out_ch),
            dtype=dtype,
        ) / jnp.sqrt(jnp.asarray(fan_in, dtype=dtype))

        b = jnp.zeros((out_ch,), dtype=dtype)
        convs.append({"W": W, "b": b})
        in_ch = out_ch

    Wout = scale * jax.random.normal(
        keys[-1],
        (in_ch, k),
        dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(in_ch, dtype=dtype))

    bout = 0.01 * jnp.arange(k, dtype=dtype)

    return {"convs": convs, "Wout": Wout, "bout": bout}


def _periodic_conv2d(x, W, b):
    kh, kw, _, _ = W.shape
    ph, pw = kh // 2, kw // 2

    xpad = jnp.pad(
        x,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="wrap",
    )

    y = jax.lax.conv_general_dilated(
        xpad,
        W,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    return y + b


def apply_cnn(params, spins, shape: tuple[int, ...]):
    dtype = _param_dtype(params)
    s = spins.astype(dtype)

    orig_batch = s.shape[:-1]
    flat = s.reshape((-1, s.shape[-1]))

    if len(shape) == 1:
        x = flat.reshape((flat.shape[0], 1, shape[0], 1))
    else:
        x = flat.reshape((flat.shape[0], shape[0], shape[1], 1))

    for layer in params["convs"]:
        x = jnp.tanh(_periodic_conv2d(x, layer["W"], layer["b"]))

    pooled = jnp.mean(x, axis=(1, 2))
    out = pooled @ params["Wout"] + params["bout"]

    return out.reshape((*orig_batch, out.shape[-1]))