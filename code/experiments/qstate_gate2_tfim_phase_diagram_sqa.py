"""`dwave-sqa` (free, local software simulator) companion to
`qstate_gate2_tfim_phase_diagram_qpu.py`, added to the same phase-diagram figure so the
paper shows simulator vs. real-hardware side by side rather than real hardware alone.

Deliberately matched to the real-QPU sweep's scale and settings, not the simulator's own
"properly-determined" `n_train=600` convention used elsewhere in this document
(`qstate_gate2_tfim_phase_anneal_sqa.py`) -- this document's own "sqa_matched" addition
already established that simulator-vs-QPU comparisons at *mismatched* settings (`num_reads`
in particular) can flip the sign of the read entirely, so this script uses the IDENTICAL
`n_train=n_val=100, n_test=200`, 3 seeds, `g~U(0,2)` phase-mixed pool, 10 g-bins, and
`num_reads=20` as `qstate_gate2_tfim_phase_diagram_qpu.py`, for a genuine apples-to-apples
overlay on the same figure. `t_anneal_us` is left at `AnnealParams`'s own default
(`qstate/anneal_arm.py::build_params` does not override it) -- NOT the real-QPU script's
`t_anneal_us=(0.5,2.0,10.0,50.0)`, which this document already found drives the simulator's
sweep count to ~1e6 and a run that should take ~2min to run 2.5+ hours instead (the same bug
documented in the "sqa_matched" addition above).

Zero QPU budget spent: `backend='dwave-sqa'` is a local CPU simulator
(`dwave.samplers.PathIntegralAnnealingSampler`), no network call, no shared ledger.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ddpm import cosine_schedule
from qstate import anneal_arm, arms, metrics, provenance, targets

OUT = Path('results/qstate')
ROOT = 20260917 + 7000  # same root family as the other tfim-phase scripts
N = 2
T = 200
NOISE_ENSEMBLE = 'hs'
J = 1.0
G_RANGE = (0.0, 2.0)
N_TRAIN, N_VAL, N_TEST = 100, 100, 200
SEEDS = range(3)
N_BINS = 10
BIN_EDGES = np.linspace(G_RANGE[0], G_RANGE[1], N_BINS + 1)
NUM_READS = 20  # matched to qstate_gate2_tfim_phase_diagram_qpu.py, not the sqa default of 50


def cell(seed):
    ds = ROOT + seed * 1000
    _, alpha_bar = cosine_schedule(T)
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

    readout_c, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], dim_r=dim_r,
                                     fidelity_fn=fid_fn)
    pred_c = readout_c.predict(c_test)

    params = anneal_arm.build_params(train['r_t'], train['t'], T, draw=seed, num_reads=NUM_READS)
    a_train = anneal_arm.anneal_pauli_features(train['r_t'], train['t'], params)
    a_val = anneal_arm.anneal_pauli_features(val['r_t'], val['t'], params)
    a_test = anneal_arm.anneal_pauli_features(test['r_t'], test['t'], params)

    readout_a, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], a_train, a_val,
                                     dim_r=dim_r, fidelity_fn=fid_fn)
    pred_a = readout_a.predict(np.column_stack([test['r_t'], a_test, te_test]))

    bin_idx = np.digitize(g_test, BIN_EDGES[1:-1])
    rows = []
    for name, pred in (('ridge_classical', pred_c), ('ridge_qrc_anneal', pred_a)):
        fid = metrics.evaluate_predictions(test['r0'], pred, N)['fidelity']
        for b in range(N_BINS):
            mask = bin_idx == b
            n_in_bin = int(mask.sum())
            if n_in_bin == 0:
                continue
            rows.append(dict(N=N, T=T, data_seed=seed, method=name, g_bin=b,
                             g_bin_center=float((BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2),
                             g_bin_low=float(BIN_EDGES[b]), g_bin_high=float(BIN_EDGES[b + 1]),
                             n_in_bin=n_in_bin, fidelity_mean=float(fid[mask].mean())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'gate2_tfim_phase_diagram_sqa.parquet'
    rows = pd.read_parquet(path).to_dict('records') if path.exists() else []
    done = {r['data_seed'] for r in rows}

    for seed in SEEDS:
        if seed in done:
            print(f'seed={seed} ja feita, pulando', flush=True)
            continue
        started = perf_counter()
        new = cell(seed)
        dt = perf_counter() - started
        rows.extend(new)
        pd.DataFrame(rows).to_parquet(path, index=False)
        d = {}
        for r in new:
            d.setdefault(r['method'], []).append(r['fidelity_mean'])
        overall_c = float(np.mean(d.get('ridge_classical', [np.nan])))
        overall_a = float(np.mean(d.get('ridge_qrc_anneal', [np.nan])))
        print(f"seed={seed} [{dt:.1f}s] classical~={overall_c:.4f} anneal~={overall_a:.4f} "
              f"(bin-wise means)", flush=True)

    frame = pd.DataFrame(rows)
    payload = dict(
        provenance=provenance.provenance(
            script='experiments/qstate_gate2_tfim_phase_diagram_sqa.py',
            seeds=dict(root=ROOT, seeds=list(SEEDS)),
            hyperparameters=dict(N=N, T=T, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                                 noise_ensemble=NOISE_ENSEMBLE, g_range=G_RANGE,
                                 n_bins=N_BINS, num_reads=NUM_READS, backend='dwave-sqa'),
            device='cpu',
        ),
        scope_note=("Free dwave-sqa simulator, g-resolved (phase-mixed pool) TFIM check, "
                    "N=2 -- matched settings (n_train/val/test, num_reads) to "
                    "qstate_gate2_tfim_phase_diagram_qpu.py's real-hardware sweep, for a "
                    "same-figure simulator-vs-QPU overlay, not the properly-determined "
                    "n_train=600 scale used elsewhere in this document."),
        n_rows=len(frame),
    )
    summary_path = OUT / 'gate2_tfim_phase_diagram_sqa_summary.json'
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path} ({len(frame)} rows) and {summary_path}\n")


if __name__ == '__main__':
    main()
