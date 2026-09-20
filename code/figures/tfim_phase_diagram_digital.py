"""Digital-reservoir-only companion to the QPU/dwave-sqa TFIM phase diagram
(`tfim_phase_diagram_qpu.py`). Same phase-mixed-pool sweep
(`experiments/qstate_gate2_tfim_phase_diagram_digital.py`), full scale
(n_train=500, n_val=500, n_test=2000, 3 seeds x 5 draws, 20 g-bins) since the digital arm
runs on a local GPU reservoir simulator, not real/simulated annealing hardware.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RED, ORANGE, MUTED, GRID = "#2a78d6", "#e34948", "#eb6834", "#52514e", "#e1e0d9"

J = 1.0
G_CROSSOVER = 1.0
FERRO_RANGE = (0.0, 0.8)
PARA_RANGE = (1.2, 2.0)


def tfim_ground_state_r(N, J, g):
    d = 2 ** N
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    def embed_1q(op, i, n):
        mats = [np.eye(2, dtype=np.complex128)] * n
        mats[i] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((d, d), dtype=np.complex128)
    for i in range(N - 1):
        H -= J * embed_1q(Z, i, N) @ embed_1q(Z, i + 1, N)
    for i in range(N):
        H -= g * embed_1q(X, i, N)
    w, v = np.linalg.eigh(H)
    psi = v[:, 0]
    rho = np.outer(psi, psi.conj())
    # Pauli-vector components 0 (IX/XI) and 14 (ZZ) for N=2, itertools.product('IXYZ', repeat=2)
    labels = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    paulis = {"I": np.eye(2, dtype=np.complex128), "X": X, "Y": np.array([[0, -1j], [1j, 0]]), "Z": Z}
    def mat(label):
        out = paulis[label[0]]
        for c in label[1:]:
            out = np.kron(out, paulis[c])
        return out
    x_exp = np.real(np.trace(rho @ mat("IX")))
    zz_exp = np.real(np.trace(rho @ mat("ZZ")))
    return x_exp, zz_exp


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
    ax.text(G_CROSSOVER, y_text, " g/J=1", color=MUTED, fontsize=8, va="bottom", ha="left", style="italic")
    ax.text(np.mean(FERRO_RANGE), y_text, "ferro", color=MUTED, fontsize=8.5, ha="center", va="bottom", style="italic")
    ax.text(np.mean(PARA_RANGE), y_text, "para", color=MUTED, fontsize=8.5, ha="center", va="bottom", style="italic")


def main():
    df = pd.read_csv("digital_appendix/tfim_phase_diagram_digital/fidelity_by_bin.csv")
    n_cells = int(df["n_cells"].iloc[0])

    g_fine = np.linspace(0.001, 2.0, 400)
    xz = np.array([tfim_ground_state_r(2, J, g) for g in g_fine])
    x_mag, zz = xz[:, 0], xz[:, 1]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 7.4), sharex=True, dpi=200,
        gridspec_kw=dict(height_ratios=[1, 1.3], hspace=0.08),
    )
    fig.patch.set_facecolor("#fcfcfb")

    style_axis(ax_top)
    ax_top.plot(g_fine, zz, color=BLUE, linewidth=2)
    ax_top.plot(g_fine, x_mag, color=RED, linewidth=2)
    ax_top.set_ylim(-1.08, 1.08)
    ax_top.set_ylabel("order parameter", color=MUTED, fontsize=9.5)
    ax_top.text(g_fine[-1], zz[-1], "  " + r"$\langle Z_1Z_2\rangle$", color=BLUE, fontsize=9, va="center")
    ax_top.text(g_fine[-1], x_mag[-1], "  " + r"$\langle X_i\rangle$", color=RED, fontsize=9, va="center")
    ax_top.set_title(
        "TFIM ground state (N=2): physical phase diagram vs.\ndigital-reservoir (QRC) reconstruction fidelity",
        color="#0b0b0b", fontsize=11.5, loc="left", pad=10,
    )
    add_phase_context(ax_top)

    style_axis(ax_bot)
    ax_bot.plot(df["bin_center"], df["classical_mean"], color=MUTED, linewidth=2, linestyle="--",
               label=f"classical (baseline) (n={n_cells} cells/bin)")
    ax_bot.fill_between(df["bin_center"], df["classical_mean"] - df["classical_sem"],
                        df["classical_mean"] + df["classical_sem"], color=MUTED, alpha=0.15, linewidth=0)
    ax_bot.plot(df["bin_center"], df["qrc_digital_mean"], color=ORANGE, linewidth=2.2, marker="o",
               markersize=5, label=f"QRC (digital reservoir) (n={n_cells} cells/bin)")
    ax_bot.fill_between(df["bin_center"], df["qrc_digital_mean"] - df["qrc_digital_sem"],
                        df["qrc_digital_mean"] + df["qrc_digital_sem"], color=ORANGE, alpha=0.15, linewidth=0)

    ax_bot.set_ylabel(f"fidelity (mean over {n_cells} cells/bin)", color=MUTED, fontsize=9.5)
    ax_bot.set_xlabel(r"transverse field $g/J$", color=MUTED, fontsize=10)
    ax_bot.set_xlim(0, 2.0)
    add_phase_context(ax_bot, label=False)
    ax_bot.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=MUTED)

    fig.text(0.5, 0.005,
            "Digital reservoir (local GPU simulator), n_train=500, n_val=500, n_test=2000, "
            "3 seeds x 5 unitary draws,\n20 g-bins. Shaded bands: ferro/para ranges used "
            "elsewhere in this repo.",
            ha="center", va="bottom", fontsize=7.3, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig("digital_appendix/tfim_phase_diagram_digital/tfim_phase_diagram_digital.png",
               dpi=200, facecolor=fig.get_facecolor())
    fig.savefig("digital_appendix/tfim_phase_diagram_digital/tfim_phase_diagram_digital.pdf",
               facecolor=fig.get_facecolor())
    print("wrote tfim_phase_diagram_digital.png/.pdf")


if __name__ == "__main__":
    main()
