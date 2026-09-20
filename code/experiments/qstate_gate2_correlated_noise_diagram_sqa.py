"""`dwave-sqa` (free, local software simulator) companion to
`qstate_gate2_correlated_noise_diagram_qpu.py`, for the same delta-resolved state-space
diagram figure. Matched settings (n_train/val/test, seeds, `num_reads=20`) to the real-QPU
sweep, same rationale as the TFIM/circular-states figures' matched simulator/QPU overlays.
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
ROOT = 20260918 + 9500  # same root family as qstate_gate2_correlated_noise_diagram_qpu.py
N = 2
T = 200
NOISE_ENSEMBLE = 'hs'
DELTA0 = np.pi / 3
N_TRAIN, N_VAL, N_TEST = 100, 100, 200
SEEDS = range(3)
N_BINS = 10
BIN_EDGES = np.linspace(-DELTA0, DELTA0, N_BINS + 1)
NUM_READS = 20  # matched to qstate_gate2_correlated_noise_diagram_qpu.py


def cell(seed):
    ds = ROOT + seed * 1000
    _, alpha_bar = cosine_schedule(T)
    train_pool, _, _ = targets.generate_correlated_noise_pool(N_TRAIN, seed=ds)
    val_pool, _, _ = targets.generate_correlated_noise_pool(N_VAL, seed=ds + 1)
    test_pool, delta_test_vals, _ = targets.generate_correlated_noise_pool(N_TEST, seed=ds + 2)

    train = arms.balanced_pairs(train_pool.r0, alpha_bar, N_TRAIN, NOISE_ENSEMBLE, ds + 10, N, T)
    val = arms.balanced_pairs(val_pool.r0, alpha_bar, N_VAL, NOISE_ENSEMBLE, ds + 110, N, T)
    test = arms.balanced_pairs(test_pool.r0, alpha_bar, N_TEST, NOISE_ENSEMBLE, ds + 210, N, T)
    delta_test = delta_test_vals[test['idx']]

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

    bin_idx = np.digitize(delta_test, BIN_EDGES[1:-1])
    rows = []
    for name, pred in (('ridge_classical', pred_c), ('ridge_qrc_anneal', pred_a)):
        fid = metrics.evaluate_predictions(test['r0'], pred, N)['fidelity']
        for b in range(N_BINS):
            mask = bin_idx == b
            n_in_bin = int(mask.sum())
            if n_in_bin == 0:
                continue
            rows.append(dict(N=N, T=T, data_seed=seed, method=name, delta_bin=b,
                             delta_bin_center=float((BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2),
                             delta_bin_low=float(BIN_EDGES[b]), delta_bin_high=float(BIN_EDGES[b + 1]),
                             n_in_bin=n_in_bin, fidelity_mean=float(fid[mask].mean())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'gate2_correlated_noise_diagram_sqa.parquet'
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
            script='experiments/qstate_gate2_correlated_noise_diagram_sqa.py',
            seeds=dict(root=ROOT, seeds=list(SEEDS)),
            hyperparameters=dict(N=N, T=T, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                                 noise_ensemble=NOISE_ENSEMBLE, delta0=DELTA0, n_bins=N_BINS,
                                 num_reads=NUM_READS, backend='dwave-sqa'),
            device='cpu',
        ),
        scope_note=("Free dwave-sqa simulator, delta-resolved correlated-noise check, N=2 -- "
                    "matched settings to qstate_gate2_correlated_noise_diagram_qpu.py's "
                    "real-hardware sweep, for a same-figure simulator-vs-QPU overlay."),
        n_rows=len(frame),
    )
    summary_path = OUT / 'gate2_correlated_noise_diagram_sqa_summary.json'
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path} ({len(frame)} rows) and {summary_path}\n")


if __name__ == '__main__':
    main()
