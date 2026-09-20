"""TFIM (N=2) phase diagram overlaid with REAL D-Wave QPU and `dwave-sqa` SIMULATOR
reconstruction fidelity, side by side, from `experiments/qstate_gate2_tfim_phase_diagram_qpu
.py` (`results/qstate/gate2_tfim_phase_diagram_qpu.parquet`) and
`qstate_gate2_tfim_phase_diagram_sqa.py` (`..._sqa.parquet`) -- the real-hardware analogue
of `tfim_phase_diagram.py`'s digital-reservoir sweep, built for the paper because a
digital-reservoir simulator run is not a substitute for real QPU data there.

Both scripts share the identical phase-mixed-pool setup, scale, seeds, and `num_reads=20`
(`n_train=n_val=100, n_test=200`, 3 seeds, 10 g-bins) so the two are a genuine apples-to-
apples overlay, not a same-order-of-magnitude comparison -- this document's own
"sqa_matched" addition found that mismatched `num_reads` between simulator and real QPU can
flip the sign of the comparison entirely. The two scripts' `ridge_classical` curves are
computed independently but from bit-identical seeds/pool/forward-noise code, so they agree
to numerical precision (confirmed in `main()` below) and only ONE is plotted. Real QPU
spend for this figure: 32.08s
(3 seeds, shared ledger `dwave_qpu_ledger_qstate_tfim_phase.json`), confirmed with the user
beforehand; the `dwave-sqa` arm is a free local CPU simulator, zero QPU budget.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qstate.targets import tfim_ground_state_r

# validated categorical slots, references/palette.md (dataviz skill)
BLUE, RED, ORANGE, AQUA, MUTED, GRID = (
    "#2a78d6", "#e34948", "#eb6834", "#1baf7a", "#52514e", "#e1e0d9",
)

J = 1.0
G_CROSSOVER = 1.0
FERRO_RANGE = (0.0, 0.8)
PARA_RANGE = (1.2, 2.0)
METHOD_STYLE = {
    "ridge_classical": dict(color=MUTED, label="classical (baseline)", ls="--", marker="none"),
    "ridge_qrc_qpu": dict(color=ORANGE, label="QRC (real D-Wave QPU)", ls="-", marker="o"),
    "ridge_qrc_anneal": dict(color=AQUA, label="QRC (dwave-sqa simulator)", ls="-", marker="s"),
}
METHOD_ORDER = ["ridge_classical", "ridge_qrc_qpu", "ridge_qrc_anneal"]


def style_axis(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def add_phase_context(ax, label=True):
    ax.axvline(G_CROSSOVER, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.axvspan(*FERRO_RANGE, color=GRID, alpha=0.5, zorder=0)
    ax.axvspan(*PARA_RANGE, color=GRID, alpha=0.5, zorder=0)
    if not label:
        return
    y0, y1 = ax.get_ylim()
    y_text = y0 + 0.04 * (y1 - y0)
    ax.text(G_CROSSOVER, y_text, " g/J=1", color=MUTED, fontsize=8, va="bottom", ha="left",
           style="italic")
    ax.text(np.mean(FERRO_RANGE), y_text, "ferro", color=MUTED, fontsize=8.5, ha="center",
           va="bottom", style="italic")
    ax.text(np.mean(PARA_RANGE), y_text, "para", color=MUTED, fontsize=8.5, ha="center",
           va="bottom", style="italic")


def main():
    df_qpu = pd.read_parquet("results/qstate/gate2_tfim_phase_diagram_qpu.parquet")
    df_sqa = pd.read_parquet("results/qstate/gate2_tfim_phase_diagram_sqa.parquet")
    n_seeds = df_qpu["data_seed"].nunique()

    # the two scripts fit ridge_classical independently from bit-identical seeds/pool code --
    # confirm agreement rather than assume it, then keep only one copy to plot.
    c_qpu = df_qpu[df_qpu["method"] == "ridge_classical"].set_index(
        ["data_seed", "g_bin_center"])["fidelity_mean"]
    c_sqa = df_sqa[df_sqa["method"] == "ridge_classical"].set_index(
        ["data_seed", "g_bin_center"])["fidelity_mean"]
    max_diff = (c_qpu - c_sqa).abs().max()
    print(f"max |classical(qpu-script) - classical(sqa-script)| = {max_diff:.2e}")
    assert max_diff < 1e-6, "the two scripts' classical arms disagree more than float noise"

    df = pd.concat([
        df_qpu[df_qpu["method"].isin(["ridge_classical", "ridge_qrc_qpu"])],
        df_sqa[df_sqa["method"] == "ridge_qrc_anneal"],
    ], ignore_index=True)

    g_fine = np.linspace(0.001, 2.0, 400)
    r_fine = np.stack([tfim_ground_state_r(2, J, g) for g in g_fine])
    zz = r_fine[:, 14]
    x_mag = r_fine[:, 0]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 7.4), sharex=True, dpi=200,
        gridspec_kw=dict(height_ratios=[1, 1.3], hspace=0.08),
    )
    fig.patch.set_facecolor("#fcfcfb")

    # --- Panel A: physical phase diagram (unchanged from the digital-arm figure) ---
    style_axis(ax_top)
    ax_top.plot(g_fine, zz, color=BLUE, linewidth=2)
    ax_top.plot(g_fine, x_mag, color=RED, linewidth=2)
    ax_top.set_ylim(-1.08, 1.08)
    ax_top.set_ylabel("order parameter", color=MUTED, fontsize=9.5)
    ax_top.text(g_fine[-1], zz[-1], "  " + r"$\langle Z_1Z_2\rangle$", color=BLUE,
               fontsize=9, va="center")
    ax_top.text(g_fine[-1], x_mag[-1], "  " + r"$\langle X_i\rangle$", color=RED,
               fontsize=9, va="center")
    ax_top.set_title(
        "TFIM ground state (N=2): physical phase diagram vs.\n"
        "D-Wave reconstruction fidelity (real QPU vs. simulator)",
        color="#0b0b0b", fontsize=11.5, loc="left", pad=10,
    )
    add_phase_context(ax_top)

    # --- Panel B: g-resolved real-QPU fidelity (n=3 seeds/bin, shown as scatter + mean) ---
    style_axis(ax_bot)
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        style = METHOD_STYLE[method]
        for g_center, seed_vals in sub.groupby("g_bin_center")["fidelity_mean"]:
            jitter = np.linspace(-0.025, 0.025, len(seed_vals))
            ax_bot.scatter(g_center + jitter, seed_vals, color=style["color"], s=12,
                          alpha=0.45, zorder=2, edgecolor="none")
        agg = (sub.groupby("g_bin_center")["fidelity_mean"]
                  .agg(["mean", "sem"]).reset_index().sort_values("g_bin_center"))
        ax_bot.plot(agg["g_bin_center"], agg["mean"], color=style["color"], linewidth=2.2,
                   linestyle=style["ls"],
                   marker=None if style["marker"] == "none" else style["marker"],
                   markersize=6, label=f"{style['label']} (n={n_seeds} seeds/bin)", zorder=3)
        ax_bot.fill_between(agg["g_bin_center"], agg["mean"] - agg["sem"],
                           agg["mean"] + agg["sem"], color=style["color"], alpha=0.15,
                           linewidth=0, zorder=1)

    ax_bot.set_ylabel(f"fidelity (mean over {n_seeds} seeds/bin)", color=MUTED, fontsize=9.5)
    ax_bot.set_xlabel(r"transverse field $g/J$", color=MUTED, fontsize=10)
    ax_bot.set_xlim(0, 2.0)
    add_phase_context(ax_bot, label=False)
    ax_bot.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=MUTED)

    fig.text(0.5, 0.005,
            "Real D-Wave QPU (Advantage_system4) vs. dwave-sqa simulator, matched settings: "
            "n_train=n_val=100, n_test=200,\n3 seeds, num_reads=20, 10 g-bins. Shaded bands: "
            "ferro/para ranges used elsewhere in this document.",
            ha="center", va="bottom", fontsize=7.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig("figures/tfim_phase_diagram_qpu.png", dpi=200, facecolor=fig.get_facecolor())
    fig.savefig("figures/tfim_phase_diagram_qpu.pdf", facecolor=fig.get_facecolor())
    print("wrote figures/tfim_phase_diagram_qpu.png and .pdf")


if __name__ == "__main__":
    main()
