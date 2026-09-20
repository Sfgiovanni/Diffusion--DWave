"""D-Wave simulated-annealing (SQA) reservoir feature extraction for Pauli-vector inputs --
the analogue of `qstate/qrc_arm.py` for the annealing arm, added on user request alongside
the digital-QRC arm already in Gate Q2 (not part of the original task specification, which
only scoped digital + Rydberg for this object).

Reuses, unmodified: `anneal.features.AnnealParams`/`anneal_features` and
`quera.encoding.fit_encoding` (both already parameter-generic -- `fit_encoding`'s internal
`_raw_channels` appends its own (sin,cos) time pair to whatever width `x_t` has, and
`AnnealParams.n_spins` is a free integer, not hardcoded to the image pipeline's 12). No
padding is needed here (unlike `qrc_arm.py`'s digital reservoir): an Ising problem has no
power-of-two qubit-tensor constraint, so `n_spins = r_t.shape[1] + 2` directly.

**Backend, and why it is safe to run alongside a concurrent D-Wave QPU job.**
`backend='dwave-sqa'` is D-Wave Ocean's own software path-integral quantum-annealing
simulator (`dwave.samplers.PathIntegralAnnealingSampler`) -- entirely local, no network call,
no shared quota. Only `backend='dwave-qpu'` (not used here) touches real hardware and the
`results/dwave_qpu_ledger.json` budget another agent's run in this session is actively using;
this module never imports `anneal.qpu_backend` or touches that ledger.

**Cost, measured directly, is why the grid here is small.** `PathIntegralAnnealingSampler`
is called once per row in a Python loop (`anneal/sqa_backend.py`), so cost is linear in row
count and steep in `n_spins`: measured on this machine, `N=1` (`n_spins=5`) costs
~0.43s/row at `num_reads=100` (~0.21s/row at 50), `N=2` (`n_spins=17`) costs ~2.85s/row at
`num_reads=100`. A full digital-arm-sized grid (500+500+2000 rows x 2 T x 3 seeds x 5 draws)
would be `~120 * 21min ~= 43h` for `N=1` alone -- infeasible in this session. Gate Q2's
anneal-arm run is therefore explicitly scoped down (see
`experiments/qstate_gate2_anneal_arm.py` and `docs/QSTATE_DIFFUSION.md`) and reported as an
exploratory, reduced-scale check, in the same spirit as `docs/AQUILA_PORT.md`'s Gate 3
preliminary Rydberg run (`samples=500` instead of `10000`) -- not a statistically powered
replacement for the digital arm's full grid.
"""
from __future__ import annotations

from anneal.features import AnnealParams, anneal_features
from quera.encoding import fit_encoding

DEFAULT_BACKEND = 'dwave-sqa'
DEFAULT_NUM_READS = 50


def build_params(r_t_train, t_train, T, draw, num_reads=DEFAULT_NUM_READS,
                 backend=DEFAULT_BACKEND, h_scale=2.0, j_scale=0.5):
    """Fit the affine encoder on TRAINING `(r_t, t)` only (per `quera/encoding.py`'s own
    "fit on train, reuse unchanged" contract) and freeze it into one `AnnealParams`, reused
    unchanged for train/val/test feature extraction -- exactly the pattern
    `qrc_fusion_fair_supervised.cell`'s Rydberg branch uses for `ReservoirParams`."""
    encoding = fit_encoding(r_t_train, t_train, T=T)
    n_spins = r_t_train.shape[1] + 2
    return AnnealParams(encoding=encoding, n_spins=n_spins, draw=draw, backend=backend,
                        num_reads=num_reads, h_scale=h_scale, j_scale=j_scale)


def anneal_pauli_features(r_t, t, params: AnnealParams):
    return anneal_features(r_t, t, params, device='cpu')
