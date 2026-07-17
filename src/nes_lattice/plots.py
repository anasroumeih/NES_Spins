from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_history(path: str | Path):
    data = _load(path)
    hist = data["history"]
    steps = np.array([h["step"] for h in hist])
    max_k = max(len(h["energies"]) for h in hist)
    energies = np.full((len(hist), max_k), np.nan)
    for a, h in enumerate(hist):
        energies[a, : len(h["energies"])] = h["energies"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(max_k):
        ax.plot(steps, energies[:, i], marker="o", label=f"NES E{i}")
    ref = hist[-1].get("reference")
    if ref is not None:
        for i, e in enumerate(ref):
            ax.axhline(e, linestyle="--", linewidth=1, label=f"ref E{i}")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("energy")
    ax.set_title(Path(path).name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_diagnostics(path: str | Path):
    data = _load(path)
    hist = data["history"]
    steps = np.array([h["step"] for h in hist])
    cond = np.array([h.get("condition_number_S", np.nan) for h in hist])
    acc = np.array([h.get("sampler_accept_rate", np.nan) for h in hist])
    trace = np.array([h.get("trace_error", np.nan) if h.get("trace_error") is not None else np.nan for h in hist])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, cond, marker="o", label="cond(S)")
    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("condition number")
    ax.set_title("Overlap conditioning")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(steps, acc, marker="o", label="NES sampler acceptance")
    if np.any(np.isfinite(trace)):
        ax2.plot(steps, trace, marker="o", label="trace error")
    ax2.set_xlabel("optimization step")
    ax2.set_title("Sampling and reference diagnostics")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    return (fig, ax), (fig2, ax2)


def print_final(path: str | Path):
    data = _load(path)
    final = data["history"][-1]
    print("File:              ", path)
    print("Config:            ", data["config"])
    print("Final NES energies:", final["energies"])
    print("Reference:         ", final.get("reference"))
    print("Reference source:  ", final.get("reference_source"))
    print("Abs errors:        ", final.get("abs_errors"))
    print("Trace error:       ", final.get("trace_error"))
    print("cond(S):           ", final.get("condition_number_S"))
    print("acceptance:        ", final.get("sampler_accept_rate"))
    print("invalid bundles:   ", final.get("invalid_bundle_fraction"))
def plot_variance(save_path):
    hist = load_history(save_path)["history"]

    steps = [h["step"] for h in hist if "state_energy_variances" in h]
    vars = [h["state_energy_variances"] for h in hist if "state_energy_variances" in h]

    vars = np.asarray(vars)          # (n_logs, k)

    fig, ax = plt.subplots()

    for i in range(vars.shape[1]):
        ax.plot(steps, vars[:, i], label=f"state {i}")

    ax.set_xlabel("training step")
    ax.set_ylabel("variance")
    ax.set_title("Energy variance during training")
    ax.legend()

    return fig, ax