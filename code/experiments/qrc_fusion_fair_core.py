"""Trimmed excerpt of the parent research repository's `experiments/qrc_fusion_fair_core.py`,
keeping only the three symbols `qstate/arms.py` needs: `LAMBDAS`, `FloorStandardizer`,
`ddim_grid` (plus the base `Standardizer` the latter extends, from the parent's
`experiments/qrc_kernel_core.py`). Everything else in the parent modules is unrelated to
the classical/QRC-digital/real-QPU/`dwave-sqa` comparison this export covers and is not
reproduced here.
"""
from __future__ import annotations

import numpy as np

T = 200
DDIM_STEPS = 50
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1e3, 1e4, 1e5)
VARIANCE_FLOOR = 1e-6


def ddim_grid(steps=DDIM_STEPS, t_max=T):
    return np.rint(np.linspace(1, t_max, steps)).astype(int)[::-1]


class Standardizer:
    def fit(self, x):
        self.mean_ = np.asarray(x, np.float64).mean(0)
        self.scale_ = np.asarray(x, np.float64).std(0)
        self.scale_[self.scale_ < 1e-10] = 1.
        return self

    def transform(self, x):
        return (np.asarray(x, np.float64) - self.mean_) / self.scale_


class FloorStandardizer(Standardizer):
    """Standardizer with a floor that actually catches float32-constant columns.

    Observables that are deterministic given a |0> memory qubit have a train-time std around
    1e-8 -- real float32 noise, not a real scale -- so a naive 1e-10 floor would pass them
    through and amplify by ~1e8. `n_floored` records how many columns were neutralised.
    """

    def fit(self, x):
        super().fit(x)
        dead = self.scale_ < VARIANCE_FLOOR
        self.n_floored = int(dead.sum())
        self.floored_ = dead
        self.scale_[dead] = 1.
        return self
