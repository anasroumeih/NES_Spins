from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .lattice import toric_code_edge_index, toric_code_terms

ModelType = Literal["ffn", "rbm", "toric_rbm", "cnn", "vit"]


@dataclass(frozen=True)
class ModelSpec:
    model: ModelType = "ffn"
    shape: tuple[int, ...] = (4, 4)
    k: int = 2
    hidden: tuple[int, ...] = (64, 64)       # FFN hidden layers
    rbm_hidden: int = 32                     # RBM hidden units per state
    channels: tuple[int, ...] = (16, 16)     # CNN channels
    kernel_size: int = 3

    # ViT / NetKet-tutorial-style Flax model parameters.
    vit_patch_size: int = 2
    vit_d_model: int = 64
    vit_num_layers: int = 2
    vit_num_heads: int = 4
    vit_mlp_ratio: int = 2
    vit_use_positional_embeddings: bool = True
    vit_log_amplitude_clip: float = 20.0

    scale: float = 0.05
    n_sites: int | None = None              # actual number of spin variables
    input_channels: int = 1                 # 2 for toric-code edge variables on a 2D cell lattice
    dtype: str = "float32"                  # keep model params/input dtype consistent

    @property
    def N(self) -> int:
        if self.n_sites is not None:
            return int(self.n_sites)
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


def _vit_kwargs(spec: ModelSpec) -> dict:
    return {
        "shape": spec.shape,
        "k": spec.k,
        "input_channels": spec.input_channels,
        "patch_size": spec.vit_patch_size,
        "d_model": spec.vit_d_model,
        "num_layers": spec.vit_num_layers,
        "num_heads": spec.vit_num_heads,
        "mlp_ratio": spec.vit_mlp_ratio,
        "use_positional_embeddings": spec.vit_use_positional_embeddings,
        "dtype": spec.dtype,
    }


def init_model(key: jax.Array, spec: ModelSpec):
    if spec.model == "ffn":
        return init_ffn(key, spec.N, spec.k, spec.hidden, spec.scale, _dtype(spec))
    if spec.model == "rbm":
        return init_rbm(key, spec.N, spec.k, spec.rbm_hidden, spec.scale, _dtype(spec))
    if spec.model == "toric_rbm":
        return init_toric_rbm(
            key,
            spec.shape,
            spec.k,
            spec.rbm_hidden,
            spec.scale,
            _dtype(spec),
        )
    if spec.model == "cnn":
        return init_cnn(
            key,
            spec.shape,
            spec.k,
            spec.channels,
            spec.kernel_size,
            spec.scale,
            _dtype(spec),
            input_channels=spec.input_channels,
        )
    if spec.model == "vit":
        from .vit import init_vit
        return init_vit(key, **_vit_kwargs(spec))
    raise ValueError(spec.model)


def apply_model(params, spins: jnp.ndarray, spec: ModelSpec) -> jnp.ndarray:
    if spec.model == "ffn":
        return apply_ffn(params, spins)
    if spec.model == "rbm":
        return apply_rbm(params, spins)
    if spec.model == "toric_rbm":
        return apply_toric_rbm(params, spins, spec.shape)
    if spec.model == "cnn":
        return apply_cnn(params, spins, spec.shape)
    if spec.model == "vit":
        from .vit import apply_vit
        return apply_vit(
            params,
            spins,
            log_amplitude_clip=spec.vit_log_amplitude_clip,
            **_vit_kwargs(spec),
        )
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
    a = scale * jax.random.normal(k1, (k, N), dtype=dtype)
    b = scale * jax.random.normal(k2, (k, n_hidden), dtype=dtype)
    W = scale * jax.random.normal(
        k3,
        (k, N, n_hidden),
        dtype=dtype,
    ) / jnp.sqrt(jnp.asarray(N, dtype=dtype))
    pref = 0.01 * jnp.arange(k, dtype=dtype)
    return {"a": a, "b": b, "W": W, "pref": pref}


def apply_rbm(params, spins):
    dtype = _param_dtype(params)
    s = spins.astype(dtype)
    visible = jnp.einsum("...n,kn->...k", s, params["a"])
    theta = jnp.einsum("...n,knh->...kh", s, params["W"]) + params["b"]
    hidden = jnp.sum(jnp.logaddexp(theta, -theta), axis=-1)
    logpsi = params["pref"] + visible + hidden
    logpsi = logpsi - jnp.log(jnp.asarray(2.0, dtype=dtype)) * params["b"].shape[-1]
    return jnp.exp(jnp.clip(logpsi, -30.0, 30.0))


# ---------- Toric-code sector RBM ----------

def init_toric_rbm(
    key,
    shape: tuple[int, ...],
    k: int,
    n_hidden: int = 32,
    scale: float = 0.03,
    dtype=jnp.float32,
):
    """RBM heads projected into toric-code flux and Wilson-loop sectors.

    This keeps the ordinary RBM untouched and only changes the support of this
    toric-specific ansatz.  Each head is assigned a Wilson sector by ``head % 4``
    and is exactly zero outside the flux-free ``B_p=+1`` manifold.
    """
    if len(shape) != 2:
        raise ValueError("toric_rbm requires a 2D toric-code shape.")
    n_edges = 2
    for L in shape:
        n_edges *= int(L)
    return init_rbm(key, n_edges, k, n_hidden, scale, dtype)


def _toric_projectors(spins: jnp.ndarray, shape: tuple[int, ...], k: int, dtype):
    """Exact flux-free and Wilson-sector projectors for flat edge spins."""
    if len(shape) != 2:
        raise ValueError("toric_rbm requires a 2D toric-code shape.")

    Lx, Ly = int(shape[0]), int(shape[1])
    _, plaquettes_np = toric_code_terms(shape, pbc=True)
    plaquettes = jnp.asarray(plaquettes_np)

    horizontal_loop = jnp.asarray(
        [toric_code_edge_index(x, 0, 0, shape) for x in range(Lx)]
    )
    vertical_loop = jnp.asarray(
        [toric_code_edge_index(0, y, 1, shape) for y in range(Ly)]
    )

    b_p = jnp.prod(spins[:, plaquettes], axis=-1)
    flux_free = jnp.prod(0.5 * (b_p + 1.0), axis=-1).astype(dtype)

    wx = jnp.prod(spins[:, horizontal_loop], axis=-1)
    wy = jnp.prod(spins[:, vertical_loop], axis=-1)

    labels = jnp.arange(k)
    target_x = (1 - 2 * (labels & 1)).astype(dtype)
    target_y = (1 - 2 * ((labels >> 1) & 1)).astype(dtype)

    sector_x = 0.5 * (wx[:, None].astype(dtype) * target_x[None, :] + 1.0)
    sector_y = 0.5 * (wy[:, None].astype(dtype) * target_y[None, :] + 1.0)
    return flux_free[:, None] * sector_x * sector_y


def apply_toric_rbm(params, spins, shape: tuple[int, ...]):
    """Apply independent RBM heads with exact toric-code sector support."""
    dtype = _param_dtype(params)
    s = spins.astype(dtype)
    orig_batch = s.shape[:-1]
    flat = s.reshape((-1, s.shape[-1]))

    rbm_vals = apply_rbm(params, flat)
    projectors = _toric_projectors(flat, shape, rbm_vals.shape[-1], dtype)
    out = rbm_vals * projectors
    return out.reshape((*orig_batch, out.shape[-1]))


# ---------- CNN ----------

def init_cnn(
    key,
    shape: tuple[int, ...],
    k: int,
    channels=(16, 16),
    kernel_size: int = 3,
    scale: float = 0.05,
    dtype=jnp.float32,
    input_channels: int = 1,
):
    if len(shape) == 1:
        if input_channels != 1:
            raise ValueError("1D CNN currently expects input_channels=1")
    elif len(shape) != 2:
        raise ValueError("CNN supports only 1D or 2D")

    keys = jax.random.split(key, len(channels) + 1)
    convs = []
    in_ch = int(input_channels)

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
    xpad = jnp.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="wrap")
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
    input_channels = params["convs"][0]["W"].shape[2]

    if len(shape) == 1:
        if input_channels != 1:
            raise ValueError("1D CNN expects one input channel")
        x = flat.reshape((flat.shape[0], 1, shape[0], 1))
    else:
        expected = shape[0] * shape[1] * input_channels
        if flat.shape[-1] != expected:
            raise ValueError(
                f"CNN expected {expected} spins for shape={shape} and input_channels={input_channels}, "
                f"got {flat.shape[-1]}."
            )
        x = flat.reshape((flat.shape[0], shape[0], shape[1], input_channels))

    for layer in params["convs"]:
        x = jnp.tanh(_periodic_conv2d(x, layer["W"], layer["b"]))

    pooled = jnp.mean(x, axis=(1, 2))
    out = pooled @ params["Wout"] + params["bout"]
    return out.reshape((*orig_batch, out.shape[-1]))
