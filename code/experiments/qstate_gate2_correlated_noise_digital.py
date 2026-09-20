"""Digital-arm comparison on QuDDPM's (arXiv:2310.05866) "correlated noise" target family
(`qstate/targets.generate_correlated_noise_pool`) -- their Table I "device characterization"
benchmark, `N=2`: a fixed clean state scattered by correlated coherent gate noise
(`e^{-i*delta*X1X2}` or `e^{-i*delta*Z1Z2}`), not a Hamiltonian curve or a geometric ring.
Structurally different from every other target family in this document -- the ensemble's
spread comes from a specific physical noise MODEL, not from this project's own forward
process or a smooth 1-parameter physical curve.

Same arms/forward/metrics/ablation discipline as `qstate_gate2_supervised.py`. Single
`T=200`, `encoding='quadrature'` -- a proportionate reduction, same as
`qstate_gate2_circular_digital.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import stats

from ddpm import cosine_schedule
from qstate import arms, metrics, provenance, qrc_arm, targets

OUT = Path('results/qstate')
ROOT = 20260918 + 1000
N = 2
T = 200
ENCODING = 'quadrature'
N_TRAIN, N_VAL, N_TEST = 500, 500, 2000
N_SEEDS = 3
N_DRAWS = 5
NOISE_ENSEMBLE = 'hs'
V_SLICES = 4


def cell(seed, draw, device):
    _, alpha_bar = cosine_schedule(T)
    ds = ROOT + seed
    train_pool, *_ = targets.generate_correlated_noise_pool(max(N_TRAIN, 100), seed=ds)
    val_pool, *_ = targets.generate_correlated_noise_pool(max(N_VAL, 100), seed=ds + 1)
    test_pool, *_ = targets.generate_correlated_noise_pool(max(N_TEST, 100), seed=ds + 2)

    train = arms.balanced_pairs(train_pool.r0, alpha_bar, N_TRAIN, NOISE_ENSEMBLE, ds + 10, N, T)
    val = arms.balanced_pairs(val_pool.r0, alpha_bar, N_VAL, NOISE_ENSEMBLE, ds + 110, N, T)
    test = arms.balanced_pairs(test_pool.r0, alpha_bar, N_TEST, NOISE_ENSEMBLE, ds + 210, N, T)

    c_train = arms.plain_design(train['r_t'], train['t'], T)
    c_val = arms.plain_design(val['r_t'], val['t'], T)
    c_test = arms.plain_design(test['r_t'], test['t'], T)
    dim_r = train['r_t'].shape[1]
    te_test = c_test[:, -10:]
    fid_fn = lambda pred, y: metrics.evaluate_predictions(y, pred, N)['fidelity']

    results = {}
    r0_hat = arms.oracle_predict(test['r_t'], test['abar'])
    results['oracle_analytic'] = dict(pred=r0_hat, n_features=0)

    c0, cs = arms.bayes_linear_moments(train['r0'], train['s'])
    r0_hat_bl = arms.bayes_linear_predict(test['r_t'], test['abar'], c0, cs)
    results['bayes_linear'] = dict(pred=r0_hat_bl, n_features=0)

    readout_c, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], dim_r=dim_r,
                                     fidelity_fn=fid_fn)
    results['ridge_classical'] = dict(pred=readout_c.predict(c_test), n_features=dim_r + 10)

    q_train, layout = qrc_arm.qrc_pauli_features(train['r_t'], train['t'], T, draw, device,
                                                 ENCODING, V_SLICES, batch=256)
    q_val, _ = qrc_arm.qrc_pauli_features(val['r_t'], val['t'], T, draw, device, ENCODING,
                                          V_SLICES, batch=256)
    q_test, _ = qrc_arm.qrc_pauli_features(test['r_t'], test['t'], T, draw, device, ENCODING,
                                           V_SLICES, batch=256)
    readout_q, *_ = arms.fit_readout(c_train, train['r0'], c_val, val['r0'], q_train, q_val,
                                     dim_r=dim_r, fidelity_fn=fid_fn)
    raw_test_q = np.column_stack([test['r_t'], q_test, te_test])
    results['ridge_qrc_digital'] = dict(pred=readout_q.predict(raw_test_q),
                                        n_features=dim_r + 10 + q_train.shape[1])


    rows = []
    for name, res in results.items():
        m = metrics.evaluate_predictions(test['r0'], res['pred'], N)
        rows.append(dict(N=N, T=T, target_kind='correlated_noise', encoding=ENCODING,
                         data_seed=seed, unitary_draw=draw, method=name, n_train=N_TRAIN,
                         n_val=N_VAL, n_test=N_TEST,
                         fidelity_mean=float(np.mean(m['fidelity'])),
                         fidelity_median=float(np.median(m['fidelity'])),
                         trace_distance_mean=float(np.mean(m['trace_distance'])),
                         nu_mean=float(np.mean(m['nu'])), nu_max=float(np.max(m['nu'])),
                         n_features=res['n_features']))
    return rows


def paired(frame, arm, baseline='ridge_classical', metric='fidelity_mean'):
    w = frame.pivot_table(index=['data_seed', 'unitary_draw'], columns='method', values=metric)
    d = (w[arm] - w[baseline]).dropna()
    if len(d) < 2:
        return None
    v = d.values
    ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
    return dict(arm=arm, baseline=baseline, n_pairs=int(len(v)), mean=float(v.mean()),
               sd=float(v.std(ddof=1)), ci95_low=float(ci[0]), ci95_high=float(ci[1]),
               p_value=float(stats.ttest_1samp(v, 0.).pvalue), wins=int((v > 0).sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    path = OUT / 'gate2_correlated_noise_digital.parquet'
    rows = pd.read_parquet(path).to_dict('records') if path.exists() else []
    done = {(r['data_seed'], r['unitary_draw']) for r in rows}

    for seed in range(N_SEEDS):
        for draw in range(N_DRAWS):
            key = (seed, draw)
            if key in done:
                continue
            started = perf_counter()
            new = cell(seed, draw, device)
            dt = perf_counter() - started
            rows.extend(new)
            pd.DataFrame(rows).to_parquet(path, index=False)
            d = {r['method']: r['fidelity_mean'] for r in new}
            b = d['ridge_classical']
            print(f"s={seed} d={draw} [{dt:.1f}s] classical={b:.4f} "
                  f"qrc={d['ridge_qrc_digital']-b:+.4f}", flush=True)

    frame = pd.DataFrame(rows)
    summary = []
    for arm in ('ridge_qrc_digital', 'bayes_linear', 'oracle_analytic'):
        s = paired(frame, arm)
        if s:
            summary.append(s)

    payload = dict(
        provenance=provenance.provenance(
            script='experiments/qstate_gate2_correlated_noise_digital.py',
            seeds=dict(root=ROOT),
            hyperparameters=dict(N=N, T=T, encoding=ENCODING, n_train=N_TRAIN, n_val=N_VAL,
                                 n_test=N_TEST, noise_ensemble=NOISE_ENSEMBLE),
            device=device,
        ),
        scope_note=("QuDDPM's 'correlated noise' target family (Table I, Fig. 5a), N=2, "
                    "digital arm. Single T/encoding -- a proportionate reduction, stated "
                    "plainly per project convention."),
        summary=summary,
    )
    summary_path = OUT / 'gate2_correlated_noise_digital_summary.json'
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path} ({len(frame)} rows) and {summary_path}\n")
    for s in summary:
        print(f"{s['arm']:18s} vs {s['baseline']:18s} delta_fid={s['mean']:+.4f} "
              f"sd={s['sd']:.4f} ci95=[{s['ci95_low']:+.4f},{s['ci95_high']:+.4f}] "
              f"p={s['p_value']:.3g} wins={s['wins']}/{s['n_pairs']}")


if __name__ == '__main__':
    main()
