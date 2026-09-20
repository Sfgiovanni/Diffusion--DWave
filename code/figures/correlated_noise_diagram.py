"""State-space diagram for QuDDPM's "correlated noise" family (`N=2`), overlaid with real
D-Wave QPU and `dwave-sqa` simulator reconstruction fidelity from `experiments/
qstate_gate2_correlated_noise_diagram_qpu.py` / `_sqa.py`.

Top panel: closed-form fidelity of the noisy state to the CLEAN state (`delta=0`) as a
function of `delta`, for each of the two coherent-error channels (`X1X2`, `Z1Z2`) --
mirrors `qstate.targets.generate_correlated_noise_pool`'s own construction
(`c=(1,1,0,1)/sqrt(3)`, `U(delta)=cos(delta)*I - i*sin(delta)*G`) but evaluated on a
deterministic fine `delta` grid rather than resampled from the noisy pool, exactly as
`tfim_phase_diagram_qpu.py` uses a clean sweep of `tfim_ground_state_r` instead of the noisy
pool for its own physical panel. This is the natural "how much does the noise perturb the
state" order parameter for this family (unlike TFIM there is no Hamiltonian ground state or
phase transition -- the noise itself, not a physical parameter, is what scatters the
ensemble).

Bottom panel: reconstruction fidelity vs `delta`, pooling both channels (matching how the
aggregate `gate2_correlated_noise_qpu.py` result elsewhere in this document does not split
by channel either). Matched settings to the real-QPU sweep (`n_train=n_val=100,n_test=200`,
3 seeds, `num_reads=20`, 10 delta-bins). Real QPU spend: shared ledger
`dwave_qpu_ledger_qstate_new_targets.json`, confirmed with the user beforehand.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RED, ORANGE, AQUA, MUTED, GRID = (
    "#2a78d6", "#e34948", "#eb6834", "#1baf7a", "#52514e", "#e1e0d9",
)

DELTA0 = np.pi / 3
METHOD_STYLE = {
    "ridge_classical": dict(color=MUTED, label="classical (baseline)", ls="--", marker="none"),
    "ridge_qrc_qpu": dict(color=ORANGE, label="QRC (real D-Wave QPU)", ls="-", marker="o"),
    "ridge_qrc_anneal": dict(color=AQUA, label="QRC (dwave-sqa simulator)", ls="-", marker="s"),
}
METHOD_ORDER = ["ridge_classical", "ridge_qrc_qpu", "ridge_qrc_anneal"]


def clean_state_fidelity(delta_grid, channel):
    """Mirrors qstate.targets.generate_correlated_noise_pool's closed form exactly, for a
    deterministic delta grid and one fixed channel, to get F(delta) = |<Psi(0)|Psi(delta)>|^2."""
    c = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.complex128)
    c = c / np.linalg.norm(c)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    G = np.kron(X, X) if channel == "xx" else np.kron(Z, Z)
    I4 = np.eye(4, dtype=np.complex128)
    psi_clean = c
    fid = np.empty_like(delta_grid)
    for i, d in enumerate(delta_grid):
        U = np.cos(d) * I4 - 1j * np.sin(d) * G
        psi_d = U @ c
        fid[i] = np.abs(np.vdot(psi_clean, psi_d)) ** 2
    return fid


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    df_qpu = pd.read_parquet("results/qstate/gate2_correlated_noise_diagram_qpu.parquet")
    df_sqa = pd.read_parquet("results/qstate/gate2_correlated_noise_diagram_sqa.parquet")
    n_seeds = df_qpu["data_seed"].nunique()

    c_qpu = df_qpu[df_qpu["method"] == "ridge_classical"].set_index(
        ["data_seed", "delta_bin_center"])["fidelity_mean"]
    c_sqa = df_sqa[df_sqa["method"] == "ridge_classical"].set_index(
        ["data_seed", "delta_bin_center"])["fidelity_mean"]
    max_diff = (c_qpu - c_sqa).abs().max()
    print(f"max |classical(qpu-script) - classical(sqa-script)| = {max_diff:.2e}")
    assert max_diff < 1e-6, "the two scripts' classical arms disagree more than float noise"

    df = pd.concat([
        df_qpu[df_qpu["method"].isin(["ridge_classical", "ridge_qrc_qpu"])],
        df_sqa[df_sqa["method"] == "ridge_qrc_anneal"],
    ], ignore_index=True)

    delta_fine = np.linspace(-DELTA0, DELTA0, 400)
    fid_xx = clean_state_fidelity(delta_fine, "xx")
    fid_zz = clean_state_fidelity(delta_fine, "zz")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, dpi=200,
        gridspec_kw=dict(height_ratios=[1, 1.3], hspace=0.08),
    )
    fig.patch.set_facecolor("#fcfcfb")

    style_axis(ax_top)
    ax_top.plot(delta_fine, fid_xx, color=BLUE, linewidth=2)
    ax_top.plot(delta_fine, fid_zz, color=RED, linewidth=2)
    ax_top.set_ylim(-0.02, 1.05)
    ax_top.set_ylabel("fidelity to clean state\n" + r"$|\langle\Psi(0)|\Psi(\delta)\rangle|^2$",
                     color=MUTED, fontsize=9)
    ax_top.text(delta_fine[-1], fid_xx[-1], "  X1X2", color=BLUE, fontsize=9, va="center")
    ax_top.text(delta_fine[-1], fid_zz[-1], "  Z1Z2", color=RED, fontsize=9, va="center")
    ax_top.set_title(
        "Correlated noise (N=2): distance from clean state vs.\n"
        "D-Wave reconstruction fidelity (real QPU vs. simulator)",
        color="#0b0b0b", fontsize=11.5, loc="left", pad=10,
    )
    ax_top.axvline(0, color=MUTED, linewidth=1, linestyle=":", zorder=1)

    style_axis(ax_bot)
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        style = METHOD_STYLE[method]
        for d_center, seed_vals in sub.groupby("delta_bin_center")["fidelity_mean"]:
            jitter = np.linspace(-0.015, 0.015, len(seed_vals))
            ax_bot.scatter(d_center + jitter, seed_vals, color=style["color"], s=12,
                          alpha=0.45, zorder=2, edgecolor="none")
        agg = (sub.groupby("delta_bin_center")["fidelity_mean"]
                  .agg(["mean", "sem"]).reset_index().sort_values("delta_bin_center"))
        ax_bot.plot(agg["delta_bin_center"], agg["mean"], color=style["color"], linewidth=2.2,
                   linestyle=style["ls"],
                   marker=None if style["marker"] == "none" else style["marker"],
                   markersize=6, label=f"{style['label']} (n={n_seeds} seeds/bin)", zorder=3)
        ax_bot.fill_between(agg["delta_bin_center"], agg["mean"] - agg["sem"],
                           agg["mean"] + agg["sem"], color=style["color"], alpha=0.15,
                           linewidth=0, zorder=1)
    ax_bot.axvline(0, color=MUTED, linewidth=1, linestyle=":", zorder=1)

    ax_bot.set_ylabel(f"fidelity (mean over {n_seeds} seeds/bin)", color=MUTED, fontsize=9.5)
    ax_bot.set_xlabel(r"correlated-error angle $\delta$ (rad)", color=MUTED, fontsize=10)
    ax_bot.set_xlim(-DELTA0, DELTA0)
    ax_bot.legend(frameon=False, fontsize=8.5, loc="lower center", labelcolor=MUTED)

    fig.text(0.5, 0.005,
            "Real D-Wave QPU (Advantage_system4) vs. dwave-sqa simulator, matched settings: "
            "n_train=n_val=100, n_test=200,\n3 seeds, num_reads=20, 10 delta-bins. Both noise "
            "channels (X1X2/Z1Z2) pooled.",
            ha="center", va="bottom", fontsize=7.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 0.93, 1))
    fig.savefig("figures/correlated_noise_diagram.png", dpi=200, facecolor=fig.get_facecolor())
    fig.savefig("figures/correlated_noise_diagram.pdf", facecolor=fig.get_facecolor())
    print("wrote figures/correlated_noise_diagram.png and .pdf")


if __name__ == "__main__":
    main()
