"""delta-resolved real-D-Wave-QPU sweep for QuDDPM's "correlated noise" family (`N=2`,
`n_spins=17`), the companion to `qstate_gate2_circular_diagram_qpu.py`/
`qstate_gate2_tfim_phase_diagram_qpu.py` for the paper's state-space diagram figures.
`generate_correlated_noise_pool` returns each sample's own error angle `delta` (and which
channel, `X1X2` or `Z1Z2`, was applied) alongside the target; `balanced_pairs`'s `idx`
indexes back into it, so fidelity is binned by `delta` after the fact, pooling both channels
(matching how the aggregate `gate2_correlated_noise_qpu.py` result elsewhere in this
document does not split by channel either).

Same scale/settings as every other real-QPU sweep in this document (`n_train=n_val=100,
n_test=200`, 3 seeds, `num_reads=20`), 10 delta-bins over `[-pi/3,pi/3]`. Reuses the
already-cached `n_spins=17` embedding and the dedicated ledger
`qstate_gate2_circular_qpu.py`/`qstate_gate2_correlated_noise_qpu.py` opened
(`dwave_qpu_ledger_qstate_new_targets.json`) -- confirmed with the user beforehand as the
second half of the combined ~85-130s estimate with the circular-states sweep (which spent
46.97s; ~144s remained going into this one).
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ddpm import cosine_schedule
from anneal.features import AnnealParams, anneal_features
from anneal.qpu_backend import QpuReservoir, set_reservoir
from anneal.qpu_budget import QpuBudget, BudgetExceeded
from qstate import arms, metrics, provenance, targets
from quera.encoding import fit_encoding

OUT = Path('results/qstate')
ROOT = 20260918 + 9500
N = 2
T = 200
NOISE_ENSEMBLE = 'hs'
DELTA0 = np.pi / 3
N_TRAIN, N_VAL, N_TEST = 100, 100, 200
SEEDS = range(3)
N_BINS = 10
BIN_EDGES = np.linspace(-DELTA0, DELTA0, N_BINS + 1)
QPU_NUM_READS = 20
QPU_T_ANNEAL_US = (0.5, 2.0, 10.0, 50.0)
QPU_BUDGET_PATH = Path('results/qstate/dwave_qpu_ledger_qstate_new_targets.json')
QPU_CAP_SECONDS = 300.0


def cell(seed, reservoir):
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

    encoding = fit_encoding(train['r_t'], train['t'], T=T)
    params = AnnealParams(encoding=encoding, n_spins=17, draw=seed, backend='dwave-qpu',
                          num_reads=QPU_NUM_READS, t_anneal_us=QPU_T_ANNEAL_US)

    a_train = anneal_features(train['r_t'], train['t'], params, device='cpu')
    print(f"    train ok, orcamento={QpuBudget(QPU_BUDGET_PATH).summary()['remaining_seconds']:.2f}s", flush=True)
    a_val = anneal_features(val['r_t'], val['t'], params, device='cpu')
    print(f"    val ok, orcamento={QpuBudget(QPU_BUDGET_PATH).summary()['remaining_seconds']:.2f}s", flush=True)
    a_test = anneal_features(test['r_t'], test['t'], params, device='cpu')
    print(f"    test ok, orcamento={QpuBudget(QPU_BUDGET_PATH).summary()['remaining_seconds']:.2f}s", flush=True)

    readout_q, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], a_train, a_val,
                                     dim_r=dim_r, fidelity_fn=fid_fn)
    pred_q = readout_q.predict(np.column_stack([test['r_t'], a_test, te_test]))

    bin_idx = np.digitize(delta_test, BIN_EDGES[1:-1])
    rows = []
    for name, pred in (('ridge_classical', pred_c), ('ridge_qrc_qpu', pred_q)):
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
    path = OUT / 'gate2_correlated_noise_diagram_qpu.parquet'
    rows = pd.read_parquet(path).to_dict('records') if path.exists() else []
    done = {r['data_seed'] for r in rows}

    budget = QpuBudget(QPU_BUDGET_PATH, cap_seconds=QPU_CAP_SECONDS, margin=0.10)
    print('orcamento (compartilhado com circular_qpu.py/correlated_noise_qpu.py/circular_diagram_qpu.py):',
         budget.summary(), flush=True)
    reservoir = QpuReservoir(n_spins=17, budget=budget, fast_anneal=False, max_tiles=3,
                             label='qstate-correlated-noise-diagram-qpu')
    print(f'n_tiles={len(reservoir.tiles)} (deve vir do cache, instantaneo)', flush=True)
    set_reservoir(reservoir)

    for seed in SEEDS:
        if seed in done:
            print(f'seed={seed} ja feita, pulando', flush=True)
            continue
        print(f'\n=== seed={seed} ===', flush=True)
        started = perf_counter()
        try:
            new = cell(seed, reservoir)
        except BudgetExceeded as exc:
            print(f'ORCAMENTO ESGOTADO, parando: {exc}', flush=True)
            break
        dt = perf_counter() - started
        rows.extend(new)
        pd.DataFrame(rows).to_parquet(path, index=False)
        d = {}
        for r in new:
            d.setdefault(r['method'], []).append(r['fidelity_mean'])
        overall_c = float(np.mean(d.get('ridge_classical', [np.nan])))
        overall_q = float(np.mean(d.get('ridge_qrc_qpu', [np.nan])))
        print(f"seed={seed} [{dt:.1f}s parede] classical~={overall_c:.4f} "
              f"qpu~={overall_q:.4f} (bin-wise means)", flush=True)

    frame = pd.DataFrame(rows)
    telemetry = reservoir.report()
    final_budget = budget.summary()
    print('\norcamento final:', final_budget, flush=True)
    print('telemetria final:', telemetry, flush=True)

    payload = dict(
        provenance=provenance.provenance(
            script='experiments/qstate_gate2_correlated_noise_diagram_qpu.py',
            seeds=dict(root=ROOT, seeds=list(SEEDS)),
            hyperparameters=dict(N=N, T=T, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                                 noise_ensemble=NOISE_ENSEMBLE, delta0=DELTA0, n_bins=N_BINS,
                                 qpu_num_reads=QPU_NUM_READS, qpu_t_anneal_us=QPU_T_ANNEAL_US,
                                 fast_anneal=False, solver=reservoir.sampler.solver.name,
                                 n_tiles=len(reservoir.tiles)),
            device='cpu',
        ),
        scope_note=("Real D-Wave QPU, delta-resolved correlated-noise check, N=2 -- the "
                    "real-hardware companion to the circular-states/TFIM g-resolved sweeps, "
                    "for the paper's state-space diagram figure. Shares the "
                    "qstate_gate2_circular_qpu.py/qstate_gate2_correlated_noise_qpu.py "
                    "dedicated ledger."),
        n_rows=len(frame),
        qpu_telemetry=telemetry,
        qpu_budget_final=final_budget,
    )
    summary_path = OUT / 'gate2_correlated_noise_diagram_qpu_summary.json'
    summary_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {path} ({len(frame)} rows) and {summary_path}\n")


if __name__ == '__main__':
    main()
