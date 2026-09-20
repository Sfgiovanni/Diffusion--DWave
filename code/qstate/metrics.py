"""Task A (reconstruction) metrics: fidelity, trace distance, purity, PSD-violation mass,
Pauli-vector MSE. Every arm's prediction goes through the identical PSD projection before
scoring -- including `oracle_analytic` -- so no arm is judged on a metric another arm was
denied the chance to fail (`docs/QSTATE_DIFFUSION.md`'s PSD-projection discipline).
"""
from __future__ import annotations

import numpy as np

from qstate.pauli import fidelity, psd_project, purity, r_to_rho, trace_distance


def evaluate_predictions(r0_true: np.ndarray, r_hat0: np.ndarray, N: int) -> dict:
    """Per-sample metrics for a batch of `(r0_true, r_hat0)` pairs.

    `r0_true` is assumed already physical (targets are constructed as genuine mixtures of
    pure states, never fit) -- `rho_true` is built without projection so a bug in target
    generation would surface as `purity_true` or a negative eigenvalue rather than being
    silently repaired. `r_hat0` always goes through `psd_project`; `nu` is the raw negative
    eigenvalue mass removed (`docs/QSTATE_DIFFUSION.md`, the honest analogue of
    `quera/encoding.py`'s `saturation_rate`).
    """
    rho_true = r_to_rho(r0_true, N)
    rho_hat_raw = r_to_rho(r_hat0, N)
    rho_hat, nu = psd_project(rho_hat_raw)
    return dict(
        fidelity=fidelity(rho_hat, rho_true),
        trace_distance=trace_distance(rho_hat, rho_true),
        purity_hat=purity(rho_hat),
        purity_true=purity(rho_true),
        nu=nu,
        pauli_mse=np.mean((r_hat0 - r0_true) ** 2, axis=1),
    )


def fidelity_to_reference(r_hat0: np.ndarray, r_ref: np.ndarray, N: int) -> np.ndarray:
    """Fidelity of each (PSD-projected) prediction against a single FIXED reference state
    (e.g. `|0><0|`'s Bloch vector), broadcast over the batch -- QuDDPM's (arXiv:2310.05866)
    own evaluation convention for its "clustered state" benchmark (`F_bar_0`, Table I):
    fidelity to the cluster CENTER, not to each sample's own true origin state. Not used
    elsewhere in this project -- `evaluate_predictions` always scores against each sample's
    own true target -- added only for the direct QuDDPM-comparison check in
    `qstate/targets.generate_clustered_pool`'s docstring.
    """
    rho_hat_raw = r_to_rho(r_hat0, N)
    rho_hat, _ = psd_project(rho_hat_raw)
    rho_ref = r_to_rho(np.tile(r_ref[None, :], (len(r_hat0), 1)), N)
    return fidelity(rho_hat, rho_ref)


def aggregate_by_t(t: np.ndarray, per_sample: dict, n_bins: int | None = None) -> list[dict]:
    """Group `evaluate_predictions`'s per-sample arrays by exact timestep value `t`.

    Metrics here are NOT safe to pool over `t` without this grouping: `oracle_analytic`'s
    Pauli MSE scales as `1/abar_t^2`, so a single pooled mean is dominated by the largest-`t`
    rows and reports a number driven by a handful of samples (`docs/QSTATE_DIFFUSION.md`).
    Fidelity is bounded in `[0,1]` and safe to pool, but is also broken out by `t` here for
    the reconstruction curve every gate script plots.
    """
    rows = []
    for tv in np.unique(t):
        mask = t == tv
        row = dict(t=int(tv), n=int(mask.sum()))
        for key, arr in per_sample.items():
            values = np.asarray(arr)[mask]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_median"] = float(np.median(values))
            row[f"{key}_sd"] = float(np.std(values, ddof=1)) if mask.sum() > 1 else 0.0
        rows.append(row)
    return rows
