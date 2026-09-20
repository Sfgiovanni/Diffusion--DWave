"""g-resolved companion to `qstate_gate2_tfim_digital.py` and
`qstate_gate2_tfim_phase_digital.py`. Those scripts report one aggregate fidelity number per
(phase or full-range) pool; this script reuses the identical phase-mixed pool
(`g ~ U(0,2)`, matching `qstate_gate2_tfim.py`'s original grid) but tracks each test row's own
`g` (via `generate_tfim_pool`'s returned `g_values`, indexed by `balanced_pairs`'s `idx`) so
fidelity can be binned as a function of `g` after the fact -- a phase diagram overlaid with
reconstruction performance, not just a single ferro/para split.

Single `T=200`/`quadrature` cell (the main grid's primary setting; `qstate_gate2_tfim_digital
.py` already showed the `T=30`/`T=200` pattern is the same and `multibase` a`quadrature`
differ only in ranking) x 3 seeds x 5 draws = 15
paired cells, `n_train=500,n_val=500,n_test=2000` -- same scale as every other Q2-family
script, so directly comparable to the numbers already published in
`docs/QSTATE_DIFFUSION.md`. Only `ridge_classical`/`ridge_qrc_digital` are computed here --
`bayes_linear`/`oracle_analytic` are omitted on purpose: both assume isotropic target/noise
statistics and are already known to catastrophically fail on the TFIM pool, which would
dominate the plot's y-scale without adding anything useful to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ddpm import cosine_schedule
from qstate import arms, metrics, provenance, qrc_arm, targets

OUT = Path('results/qstate')
ROOT = 20260917 + 8000
N = 2
T = 200
ENCODING = 'quadrature'
N_TRAIN, N_VAL, N_TEST = 500, 500, 2000
N_SEEDS = 3
N_DRAWS = 5
NOISE_ENSEMBLE = 'hs'
V_SLICES = 4
J = 1.0
G_RANGE = (0.0, 2.0)
N_BINS = 20
BIN_EDGES = np.linspace(G_RANGE[0], G_RANGE[1], N_BINS + 1)


def batch_for(N):
    return 1024 if N == 1 else 256


def cell(seed, draw, device):
    _, alpha_bar = cosine_schedule(T)
    ds = ROOT + seed
    train_pool, _ = targets.generate_tfim_pool(N, N_TRAIN, seed=ds, J=J, g_range=G_RANGE)
    val_pool, _ = targets.generate_tfim_pool(N, N_VAL, seed=ds + 1, J=J, g_range=G_RANGE)
    test_pool, g_test_vals = targets.generate_tfim_pool(N, N_TEST, seed=ds + 2, J=J,
                                                        g_range=G_RANGE)

    train = arms.balanced_pairs(train_pool.r0, alpha_bar, N_TRAIN, NOISE_ENSEMBLE, ds + 10, N, T)
    val = arms.balanced_pairs(val_pool.r0, alpha_bar, N_VAL, NOISE_ENSEMBLE, ds + 110, N, T)
    test = arms.balanced_pairs(test_pool.r0, alpha_bar, N_TEST, NOISE_ENSEMBLE, ds + 210, N, T)
    g_test = g_test_vals[test['idx']]

    c_train = arms.plain_design(train['r_t'], train['t'], T)
    c_val = arms.plain_design(val['r_t'], val['t'], T)
    c_test = arms.plain_design(test['r_t'], test['t'], T)
    dim_r = train['r_t'].shape[1]
    te_test = c_test[:, -10:]
    fid_fn = lambda pred, y: metrics.evaluate_predictions(y, pred, N)['fidelity']

    preds = {}
    readout_c, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], dim_r=dim_r,
                                     fidelity_fn=fid_fn)
    preds['ridge_classical'] = readout_c.predict(c_test)

    q_train, layout = qrc_arm.qrc_pauli_features(train['r_t'], train['t'], T, draw, device,
                                                 ENCODING, V_SLICES, batch=batch_for(N))
    q_val, _ = qrc_arm.qrc_pauli_features(val['r_t'], val['t'], T, draw, device, ENCODING,
                                          V_SLICES, batch=batch_for(N))
    q_test, _ = qrc_arm.qrc_pauli_features(test['r_t'], test['t'], T, draw, device, ENCODING,
                                           V_SLICES, batch=batch_for(N))
    readout_q, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], q_train, q_val,
                                     dim_r=dim_r, fidelity_fn=fid_fn)
    preds['ridge_qrc_digital'] = readout_q.predict(np.column_stack([test['r_t'], q_test, te_test]))

    bin_idx = np.digitize(g_test, BIN_EDGES[1:-1])
    rows = []
    for name, pred in preds.items():
        fid = metrics.evaluate_predictions(test['r0'], pred, N)['fidelity']
        for b in range(N_BINS):
            mask = bin_idx == b
            n_in_bin = int(mask.sum())
            if n_in_bin == 0:
                continue
            rows.append(dict(N=N, T=T, encoding=ENCODING, data_seed=seed, unitary_draw=draw,
                             method=name, g_bin=b,
                             g_bin_center=float((BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2),
                             g_bin_low=float(BIN_EDGES[b]), g_bin_high=float(BIN_EDGES[b + 1]),
                             n_in_bin=n_in_bin, fidelity_mean=float(fid[mask].mean())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    path = OUT / 'gate2_tfim_phase_diagram.parquet'
    rows = pd.read_parquet(path).to_dict('records') if path.exists() else []
    done = {(r['data_seed'], r['unitary_draw']) for r in rows}

    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            if (seed, draw) in done:
                continue
            started = perf_counter()
            new = cell(seed, draw, device)
            dt = perf_counter() - started
            rows.extend(new)
            pd.DataFrame(rows).to_parquet(path, index=False)
            print(f"s={seed} d={draw} [{dt:.1f}s] {len(new)} bin-rows", flush=True)

    frame = pd.DataFrame(rows)
    payload = dict(
        provenance=provenance.provenance(
            script='experiments/qstate_gate2_tfim_phase_diagram.py',
            seeds=dict(root=ROOT),
            hyperparameters=dict(N=N, T=T, encoding=ENCODING, n_train=N_TRAIN, n_val=N_VAL,
                                 n_test=N_TEST, noise_ensemble=NOISE_ENSEMBLE, J=J,
                                 g_range=G_RANGE, n_bins=N_BINS, n_seeds=N_SEEDS,
                                 n_draws=N_DRAWS),
            device=device,
        ),
        scope_note=("g-resolved fidelity for the phase-mixed TFIM pool (g~U(0,2)), binned "
                    "per test row's own g via generate_tfim_pool's g_values and "
                    "balanced_pairs' idx, to overlay reconstruction performance on the "
                    "physical order-parameter phase diagram rather than reporting only the "
                    "phase-mixed or ferro/para-split aggregates."),
        n_rows=len(frame),
    )
    summary_path = OUT / 'gate2_tfim_phase_diagram_summary.json'
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path} ({len(frame)} rows) and {summary_path}\n")


if __name__ == '__main__':
    main()
