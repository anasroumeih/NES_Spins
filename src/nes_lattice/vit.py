"""Flax Vision-Transformer ansatz for the NES lattice project.

This module is a compact, multi-output adaptation of the architecture used in
NetKet's Vision-Transformer wave-function tutorial.  NetKet currently presents
that ViT as tutorial Flax code rather than as a stable ``nk.models.ViT`` class;
we therefore keep the module locally and expose it through the project's
``init_model`` / ``apply_model`` interface.

The model is intentionally real and positive after conversion from log
amplitudes.  That is appropriate for the stoquastic TFIM and toric-code
Hamiltonians in this project.  The rest of the code still sees ordinary real
amplitudes ``psi_i(sigma)`` with one output per NES state.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def _as_dtype(name: str):
    return jnp.dtype(name)


def _extract_patches_2d(
    x: jnp.ndarray,
    *,
    shape: tuple[int, int],
    input_channels: int,
    patch_size: int,
) -> jnp.ndarray:
    """Convert flat spin configurations to a batch of flattened image patches.

    Parameters
    ----------
    x:
        Array of shape ``(batch, Lx*Ly*input_channels)``.
    shape:
        ``(Lx, Ly)`` cell lattice.  Toric code uses two channels per cell.
    input_channels:
        One for TFIM/Heisenberg, two for the toric-code horizontal/vertical
        edge variables.
    patch_size:
        Linear side length of a non-overlapping square patch.
    """
    Lx, Ly = shape
    if Lx % patch_size != 0 or Ly % patch_size != 0:
        raise ValueError(
            f"ViT patch_size={patch_size} must divide both lattice dimensions "
            f"shape={shape}."
        )

    expected = Lx * Ly * input_channels
    if x.shape[-1] != expected:
        raise ValueError(
            f"ViT expected {expected} spin variables for shape={shape} and "
            f"input_channels={input_channels}, got {x.shape[-1]}."
        )

    batch = x.shape[0]
    nx, ny = Lx // patch_size, Ly // patch_size
    x = x.reshape(batch, Lx, Ly, input_channels)
    x = x.reshape(batch, nx, patch_size, ny, patch_size, input_channels)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(batch, nx * ny, patch_size * patch_size * input_channels)


class _TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Pre-layer-normalization residual transformer block.
        y = nn.LayerNorm(param_dtype=self.param_dtype, dtype=x.dtype, name="ln_attn")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
            param_dtype=self.param_dtype,
            dtype=x.dtype,
            name="attention",
        )(y, y)
        x = x + y

        y = nn.LayerNorm(param_dtype=self.param_dtype, dtype=x.dtype, name="ln_mlp")(x)
        y = nn.Dense(
            self.mlp_ratio * self.d_model,
            param_dtype=self.param_dtype,
            dtype=x.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="mlp_in",
        )(y)
        y = nn.gelu(y)
        y = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            dtype=x.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="mlp_out",
        )(y)
        return x + y


class NESVisionTransformer(nn.Module):
    """A 2D ViT with ``k`` real log-amplitude heads for NES.

    The module returns ``log_psi`` of shape ``(batch, k)``.  The project-level
    wrapper converts those values to positive amplitudes using a clipped
    exponential, keeping the rest of the NES code unchanged.
    """

    shape: tuple[int, int]
    k: int
    input_channels: int = 1
    patch_size: int = 2
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 2
    use_positional_embeddings: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, spins: jnp.ndarray) -> jnp.ndarray:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"vit_d_model={self.d_model} must be divisible by "
                f"vit_num_heads={self.num_heads}."
            )

        x = jnp.atleast_2d(spins).astype(self.param_dtype)
        patches = _extract_patches_2d(
            x,
            shape=self.shape,
            input_channels=self.input_channels,
            patch_size=self.patch_size,
        )
        n_patches = patches.shape[1]

        x = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="patch_embed",
        )(patches)

        if self.use_positional_embeddings:
            pos = self.param(
                "pos_embedding",
                nn.initializers.normal(stddev=0.02),
                (1, n_patches, self.d_model),
                self.param_dtype,
            )
            x = x + pos

        for layer in range(self.num_layers):
            x = _TransformerBlock(
                d_model=self.d_model,
                n_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                name=f"encoder_{layer}",
            )(x)

        # NetKet's tutorial pools the patch sequence before its output head.
        # Mean pooling keeps the log-amplitude scale less sensitive to system size.
        x = jnp.mean(x, axis=1)
        x = nn.LayerNorm(param_dtype=self.param_dtype, dtype=x.dtype, name="final_norm")(x)
        return nn.Dense(
            self.k,
            param_dtype=self.param_dtype,
            dtype=x.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name="nes_heads",
        )(x)


def make_vit_module(
    *,
    shape: tuple[int, ...],
    k: int,
    input_channels: int,
    patch_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    mlp_ratio: int,
    use_positional_embeddings: bool,
    dtype: str,
) -> NESVisionTransformer:
    if len(shape) != 2:
        raise ValueError(
            "The ViT ansatz is implemented for 2D lattices only. "
            "Use FFN/RBM/CNN for the project's 1D exception."
        )
    if patch_size < 1:
        raise ValueError("vit_patch_size must be at least one.")
    if d_model < 1 or num_layers < 1 or num_heads < 1 or mlp_ratio < 1:
        raise ValueError("ViT dimensions/layer/head counts must be positive.")
    if d_model % num_heads != 0:
        raise ValueError("vit_d_model must be divisible by vit_num_heads.")
    return NESVisionTransformer(
        shape=(int(shape[0]), int(shape[1])),
        k=int(k),
        input_channels=int(input_channels),
        patch_size=int(patch_size),
        d_model=int(d_model),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        mlp_ratio=int(mlp_ratio),
        use_positional_embeddings=bool(use_positional_embeddings),
        param_dtype=_as_dtype(dtype),
    )


def init_vit(
    key: jax.Array, **kwargs):
    module = make_vit_module(**kwargs)
    Lx, Ly = kwargs["shape"]
    n = int(Lx) * int(Ly) * int(kwargs["input_channels"])
    dummy = jnp.ones((1, n), dtype=_as_dtype(kwargs["dtype"]))
    return module.init(key, dummy)


def apply_vit(
    params,
    spins: jnp.ndarray,
    *,
    log_amplitude_clip: float,
    **kwargs,
) -> jnp.ndarray:
    """Return real NES amplitudes of shape ``(..., k)``.

    Existing project code works with amplitudes, whereas the NetKet ViT
    tutorial naturally produces log-amplitudes.  Exponentiating here is the
    adapter layer.  The clip prevents avoidable overflow, not a modification
    of the NES determinant itself.
    """
    module = make_vit_module(**kwargs)
    dtype = _as_dtype(kwargs["dtype"])
    x = spins.astype(dtype)
    original_batch_shape = x.shape[:-1]
    flat = x.reshape((-1, x.shape[-1]))
    logpsi = module.apply(params, flat)
    logpsi = jnp.clip(logpsi, -float(log_amplitude_clip), float(log_amplitude_clip))
    psi = jnp.exp(logpsi)
    return psi.reshape((*original_batch_shape, psi.shape[-1]))
