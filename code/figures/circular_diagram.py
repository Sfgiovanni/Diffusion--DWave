"""State-space diagram for QuDDPM's "circular states" family (`N=1`), the geometric analogue
of `tfim_phase_diagram_qpu.py`'s physical phase diagram, overlaid with real D-Wave QPU and
`dwave-sqa` simulator reconstruction fidelity from `experiments/
qstate_gate2_circular_diagram_qpu.py` / `_sqa.py`.

Unlike TFIM, `|psi_x> = cos(x)|0> + sin(x)|1>` (`x~U[0,2*pi)`) has no Hamiltonian and no
phase transition -- `docs/QSTATE_DIFFUSION.md` is explicit this is a purely geometric,
isotropic-around-the-circle family. `<Z>=cos(2x)`, `<X>=sin(2x)`, `<Y>=0` exactly (the state
lives on the Bloch sphere's X-Z great circle; period `pi` in `x`, so the circle is traced
twice as `x` runs over `[0,2*pi)` -- a property of the sampling convention, not a bug). This
figure checks a different question than TFIM's: is reconstruction fidelity uniform around
the circle, or does the encoding/reservoir's own basis choice break that geometric symmetry?

Both backend scripts share scale/seeds/`num_reads=20` (matched, not just same-order-of-
magnitude) -- `n_train=n_val=100, n_test=200`, 3 seeds, 12 x-bins (30 degrees each). Real QPU
spend: 46.97s (shared ledger `dwave_qpu_ledger_qstate_new_targets.json`), confirmed with the
user beforehand; `dwave-sqa` is a free local CPU simulator.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RED, ORANGE, AQUA, MUTED, GRID = (
    "#2a78d6", "#e34948", "#eb6834", "#1baf7a", "#52514e", "#e1e0d9",
)

METHOD_STYLE = {
    "ridge_classical": dict(color=MUTED, label="classical (baseline)", ls="--", marker="none"),
    "ridge_qrc_qpu": dict(color=ORANGE, label="QRC (real D-Wave QPU)", ls="-", marker="o"),
    "ridge_qrc_anneal": dict(color=AQUA, label="QRC (dwave-sqa simulator)", ls="-", marker="s"),
}
METHOD_ORDER = ["ridge_classical", "ridge_qrc_qpu", "ridge_qrc_anneal"]
PI_TICKS = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
PI_LABELS = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    df_qpu = pd.read_parquet("results/qstate/gate2_circular_diagram_qpu.parquet")
    df_sqa = pd.read_parquet("results/qstate/gate2_circular_diagram_sqa.parquet")
    n_seeds = df_qpu["data_seed"].nunique()

    c_qpu = df_qpu[df_qpu["method"] == "ridge_classical"].set_index(
        ["data_seed", "x_bin_center"])["fidelity_mean"]
    c_sqa = df_sqa[df_sqa["method"] == "ridge_classical"].set_index(
        ["data_seed", "x_bin_center"])["fidelity_mean"]
    max_diff = (c_qpu - c_sqa).abs().max()
    print(f"max |classical(qpu-script) - classical(sqa-script)| = {max_diff:.2e}")
    assert max_diff < 1e-6, "the two scripts' classical arms disagree more than float noise"

    df = pd.concat([
        df_qpu[df_qpu["method"].isin(["ridge_classical", "ridge_qrc_qpu"])],
        df_sqa[df_sqa["method"] == "ridge_qrc_anneal"],
    ], ignore_index=True)

    x_fine = np.linspace(0, 2 * np.pi, 400)
    z_exp = np.cos(2 * x_fine)
    x_exp = np.sin(2 * x_fine)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, dpi=200,
        gridspec_kw=dict(height_ratios=[1, 1.3], hspace=0.08),
    )
    fig.patch.set_facecolor("#fcfcfb")

    style_axis(ax_top)
    ax_top.plot(x_fine, z_exp, color=BLUE, linewidth=2)
    ax_top.plot(x_fine, x_exp, color=RED, linewidth=2)
    ax_top.set_ylim(-1.08, 1.08)
    ax_top.set_ylabel(r"Bloch coordinate ($\langle Z\rangle=\cos 2x$, $\langle X\rangle=\sin 2x$)",
                     color=MUTED, fontsize=8.5)
    ax_top.text(x_fine[-1], z_exp[-1], "  " + r"$\langle Z\rangle$", color=BLUE,
               fontsize=10, va="center")
    ax_top.text(x_fine[-1], x_exp[-1], "  " + r"$\langle X\rangle$", color=RED,
               fontsize=10, va="bottom")
    ax_top.set_title(
        "Circular states (N=1): position on the Bloch circle vs.\n"
        "D-Wave reconstruction fidelity (real QPU vs. simulator)",
        color="#0b0b0b", fontsize=11.5, loc="left", pad=10,
    )

    style_axis(ax_bot)
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        style = METHOD_STYLE[method]
        for x_center, seed_vals in sub.groupby("x_bin_center")["fidelity_mean"]:
            jitter = np.linspace(-0.06, 0.06, len(seed_vals))
            ax_bot.scatter(x_center + jitter, seed_vals, color=style["color"], s=12,
                          alpha=0.45, zorder=2, edgecolor="none")
        agg = (sub.groupby("x_bin_center")["fidelity_mean"]
                  .agg(["mean", "sem"]).reset_index().sort_values("x_bin_center"))
        ax_bot.plot(agg["x_bin_center"], agg["mean"], color=style["color"], linewidth=2.2,
                   linestyle=style["ls"],
                   marker=None if style["marker"] == "none" else style["marker"],
                   markersize=6, label=f"{style['label']} (n={n_seeds} seeds/bin)", zorder=3)
        ax_bot.fill_between(agg["x_bin_center"], agg["mean"] - agg["sem"],
                           agg["mean"] + agg["sem"], color=style["color"], alpha=0.15,
                           linewidth=0, zorder=1)

    ax_bot.set_ylabel(f"fidelity (mean over {n_seeds} seeds/bin)", color=MUTED, fontsize=9.5)
    ax_bot.set_xlabel(r"Bloch angle $x$ (rad)", color=MUTED, fontsize=10)
    ax_bot.set_xlim(0, 2 * np.pi)
    ax_bot.set_xticks(PI_TICKS)
    ax_bot.set_xticklabels(PI_LABELS)
    ax_bot.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=1, labelcolor=MUTED)

    fig.text(0.5, 0.005,
            "Real D-Wave QPU (Advantage_system4) vs. dwave-sqa simulator, matched settings: "
            "n_train=n_val=100, n_test=200,\n3 seeds, num_reads=20, 12 x-bins. Purely "
            "geometric family (no phase transition).",
            ha="center", va="bottom", fontsize=7.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 0.93, 1))
    fig.savefig("figures/circular_diagram.png", dpi=200, facecolor=fig.get_facecolor())
    fig.savefig("figures/circular_diagram.pdf", facecolor=fig.get_facecolor())
    print("wrote figures/circular_diagram.png and .pdf")


if __name__ == "__main__":
    main()
