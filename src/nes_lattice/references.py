from __future__ import annotations

import numpy as np

from .hamiltonians import HamiltonianSpec
from .evaluation import own_ed_reference


def netket_reference_energies(hspec: HamiltonianSpec, k: int, max_states: int = 2_000_000):
    """Optional NetKet Lanczos/exact reference.

    This is intentionally optional: the project still runs without NetKet. For
    larger systems, NetKet's sparse operators/Lanczos are much more appropriate
    than the tiny dense ED fallback in this project.
    """
    try:
        import netket as nk
    except Exception as exc:  # pragma: no cover
        return None, f"NetKet unavailable: {type(exc).__name__}: {exc}"

    try:
        shape = hspec.shape
        if len(shape) == 1:
            graph = nk.graph.Chain(length=shape[0], pbc=hspec.pbc)
        elif len(shape) == 2:
            graph = nk.graph.Grid(extent=shape, pbc=hspec.pbc)
        else:
            return None, f"NetKet reference supports 1D/2D only, got {shape}"

        if hspec.name == "tfim":
            hi = nk.hilbert.Spin(s=0.5, N=hspec.N)
            # NetKet Ising convention: H = -J sum sz_i sz_j - h sum sx_i
            op = nk.operator.Ising(hilbert=hi, graph=graph, h=hspec.g, J=hspec.J)
        elif hspec.name == "heisenberg":
            if hspec.magnetization == 0:
                hi = nk.hilbert.Spin(s=0.5, N=hspec.N, total_sz=0)
            else:
                hi = nk.hilbert.Spin(s=0.5, N=hspec.N)
            op = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=hspec.J)
        else:
            return None, f"unknown Hamiltonian {hspec.name}"

        if hi.n_states > max_states:
            return None, f"NetKet ED skipped because hilbert size {hi.n_states} > max_states={max_states}"

        vals = nk.exact.lanczos_ed(op, k=k, compute_eigenvectors=False)
        vals = np.asarray(vals, dtype=np.float64)
        vals.sort()
        return vals[:k], "netket_lanczos_ed"
    except Exception as exc:  # pragma: no cover
        return None, f"NetKet reference failed: {type(exc).__name__}: {exc}"


def get_reference_energies(hspec: HamiltonianSpec, k: int, prefer: str = "auto", own_ed_max_sites: int = 14, netket_max_states: int = 2_000_000):
    """Return (energies_or_None, source_message)."""
    prefer = prefer.lower()
    if prefer in ("netket", "auto"):
        vals, msg = netket_reference_energies(hspec, k, max_states=netket_max_states)
        if vals is not None:
            return vals, msg
        if prefer == "netket":
            return None, msg
    if prefer in ("ed", "own_ed", "auto"):
        vals, msg = own_ed_reference(hspec, k, max_sites=own_ed_max_sites)
        if vals is not None:
            return vals, msg
        return None, msg
    if prefer == "none":
        return None, "reference disabled"
    return None, f"unknown reference mode: {prefer}"
