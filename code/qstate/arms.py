"""Readout arms for Task A (denoising `r0` from `r_t`), and the pairing/design machinery
they share.

Reuses, unchanged, from `experiments.qrc_fusion_fair_core`: `FloorStandardizer`, `LAMBDAS`,
`ddim_grid`
(depends only on `T`, not on the latent's meaning). Does NOT reuse that module's `Readout`
or `fit_readout`: both hardcode `raw[:, :10]`/`raw[:, -10:]` slices for the 10-dim image
latent, which would silently mis-slice a 3- or 15-dim Pauli vector. `StateReadout`/
`fit_readout` below are the same ridge-over-[standardized-classical|standardized-extra]
design mirrored with the block width as a parameter instead of a literal `10`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge

from denoiser_classical import sinusoidal_embedding
from experiments.qrc_fusion_fair_core import LAMBDAS, FloorStandardizer, ddim_grid
from qstate import forward, qrc_arm

DDIM_STEPS_SUBGRID = 50


def grid_steps_for(T: int) -> int:
    """`T` timesteps balanced over a `min(50, T)`-point subgrid -- for `T=200` this is the
    same 50-point DDIM subgrid `qrc_fusion_fair_core.balanced_pairs` uses; for `T=30` (the
    QGDM paper's own schedule length) there is no need to subsample, so all 30 steps are
    used directly rather than padding a 50-point request onto a 30-length range."""
    return min(DDIM_STEPS_SUBGRID, T)


def balanced_pairs(r0_pool: np.ndarray, alpha_bar: np.ndarray, count: int, noise_ensemble: str,
                   seed: int, N: int, T: int, grid_steps: int | None = None) -> dict:
    """Pairs balanced over the `t` grid, mirroring `qrc_fusion_fair_core.balanced_pairs`'s
    protocol exactly except for the noise model: `epsilon ~ N(0,1)` there becomes
    `s ~ sample_noise(noise_ensemble)` here, per `docs/QSTATE_DIFFUSION.md`'s forward.
    """
    rng = np.random.default_rng(seed)
    steps = grid_steps or grid_steps_for(T)
    t = np.resize(ddim_grid(steps, T), count)
    rng.shuffle(t)
    idx = rng.choice(len(r0_pool), count, replace=count > len(r0_pool))
    r0 = r0_pool[idx]
    r_t, s, abar = forward.forward_sample(r0, t, alpha_bar, N, noise_ensemble, rng)
    return dict(r_t=r_t, r0=r0, s=s, t=t, idx=idx, abar=abar)


def plain_design(r: np.ndarray, t: np.ndarray, T: int) -> np.ndarray:
    """`[r_t | te(t)]`, `te` the same 10-dim sinusoidal timestep embedding the image
    pipeline uses (`denoiser_classical.sinusoidal_embedding`)."""
    return np.column_stack([r, sinusoidal_embedding(torch.as_tensor(t), 10, T).numpy()])


def interaction_extra(plain: np.ndarray, dim_r: int) -> np.ndarray:
    """`r_t (x) te(t)` outer product, flattened -- the structured classical control."""
    r, te = plain[:, :dim_r], plain[:, -10:]
    return (r[:, :, None] * te[:, None, :]).reshape(len(r), -1)


@dataclass
class StateReadout:
    """Ridge over `[standardized [r_t|te] | standardized extra]`.

    `raw` is always laid out `[r_t (dim_r) | extra (width, optional) | te (10)]`, matching
    how every arm below builds its design matrix.
    """
    classical: FloorStandardizer
    extra: FloorStandardizer | None
    model: Ridge
    dim_r: int

    def design(self, raw: np.ndarray) -> np.ndarray:
        c = np.column_stack([raw[:, :self.dim_r], raw[:, -10:]])
        if self.extra is None:
            return self.classical.transform(c)
        e = raw[:, self.dim_r:-10]
        return np.column_stack([self.classical.transform(c), self.extra.transform(e)])

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.model.predict(self.design(raw))


def fit_readout(c_train, y_train, c_val, y_val, e_train=None, e_val=None, dim_r=None,
                fidelity_fn=None):
    """Lambda selected on validation MSE (matches `qrc_fusion_fair_core.fit_readout`, keeps
    the selection rule comparable across objects). `fidelity_fn(pred_val, y_val) -> array`,
    if given, is additionally evaluated at every lambda and returned in `curve` -- this
    project's best-established pattern is MSE/fidelity dissociation, so recording both
    at near-zero cost makes it visible instead of hidden (see `docs/QSTATE_DIFFUSION.md`).
    """
    dim_r = dim_r if dim_r is not None else c_train.shape[1] - 10
    cs = FloorStandardizer().fit(c_train)
    es = None
    if e_train is None:
        dt, dv = cs.transform(c_train), cs.transform(c_val)
    else:
        es = FloorStandardizer().fit(e_train)
        dt = np.column_stack([cs.transform(c_train), es.transform(e_train)])
        dv = np.column_stack([cs.transform(c_val), es.transform(e_val)])
    best = None
    curve = []
    for lam in LAMBDAS:
        m = Ridge(alpha=lam).fit(dt, y_train)
        pred_val = m.predict(dv)
        mse = float(np.mean((pred_val - y_val) ** 2))
        fid = float(np.mean(fidelity_fn(pred_val, y_val))) if fidelity_fn is not None else None
        curve.append(dict(lam=float(lam), val_mse=mse, val_fidelity=fid))
        if best is None or mse < best[0]:
            best = (mse, float(lam), m)
    readout = StateReadout(cs, es, best[2], dim_r)
    n_floored = es.n_floored if es is not None else 0
    return readout, best[0], best[1], curve, n_floored


def oracle_predict(r_t: np.ndarray, abar: np.ndarray) -> np.ndarray:
    """`r_hat_0 = r_t / abar_t`, the exact closed-form inverse of the QGDM *deterministic*
    forward (Eq. 2). Under the stochastic forward this is `r0 + ((1-abar)/abar) s`: unbiased
    but its error grows as `abar -> 0`, which is exactly the non-triviality Q1 checks for.
    """
    return r_t / abar[:, None]


def bayes_linear_moments(r0: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """`(c0, cs)`: per-component second moments `E|r0|^2 / K`, `E|s|^2 / K`, estimated by
    Monte Carlo over a batch of training pairs' `r0` and matched noise draws `s`."""
    K = r0.shape[1]
    c0 = float((r0 ** 2).sum(1).mean()) / K
    cs = float((s ** 2).sum(1).mean()) / K
    return c0, cs


def bayes_linear_predict(r_t: np.ndarray, abar: np.ndarray, c0: float, cs: float) -> np.ndarray:
    """The Bayes-optimal *linear*, *time-dependent-scalar* estimator `r_hat_0 = a*(abar) r_t`
    under isotropic target and noise ensembles (`docs/QSTATE_DIFFUSION.md` degeneracy 2):
    `a*(abar) = abar*c0 / (abar^2*c0 + (1-abar)^2*cs)`. Not a competing arm -- the analytic
    ceiling that makes a ridge-arm tie interpretable (all three at the ceiling vs. all three
    merely tied with each other)."""
    a_star = abar * c0 / (abar ** 2 * c0 + (1.0 - abar) ** 2 * cs + 1e-300)
    return a_star[:, None] * r_t
