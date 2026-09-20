"""Stochastic forward channel for the quantum-state diffusion object.

QGDM's forward (arXiv:2401.07039, Eq. 2) is `rho_t = (1-abar_t) I/d + abar_t rho_0`: a
deterministic scalar shrinkage of the Pauli vector, `r_t = abar_t r_0`. That map is
exactly invertible (`r_0 = r_t / abar_t` for any `abar_t > 0`), which makes it a trivial
regression target for any readout -- see `docs/QSTATE_DIFFUSION.md` degeneracy (1). This
module replaces it with the marginal `rho_t = (1-abar_t) sigma + abar_t rho_0`,
`sigma ~ E`, `E[sigma] = I/d`, which recovers the QGDM channel exactly in expectation
(Q0 checks this against Eq. 2 by Monte Carlo) but is not invertible from a single draw.

In Pauli-vector form: `r_t = abar_t r_0 + (1-abar_t) s`, `s` = Bloch vector of `sigma`,
`E[s] = 0`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magicqrc.circuits import haar_random_unitary
from qstate.pauli import pauli_basis, rho_to_r

NOISE_ENSEMBLES = ('hs', 'haar', 'pauli_twirl')


def _hs_noise(N: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """Hilbert-Schmidt-induced random density matrix: `rho = G G^dagger / tr(G G^dagger)`
    for `G` a complex Ginibre matrix. Unitarily invariant, so `E[rho] = I/d` exactly; the
    ensemble mean of `|s|^2` is checked against `(d^2-1)/(d^2+1)` by Monte Carlo in
    `tests/test_qstate_forward.py` rather than hardcoded as ground truth."""
    d = 2 ** N
    g = (rng.normal(size=(batch, d, d)) + 1j * rng.normal(size=(batch, d, d))) / np.sqrt(2.0)
    rho = g @ np.conj(np.transpose(g, (0, 2, 1)))
    rho /= np.trace(rho, axis1=1, axis2=2).real[:, None, None]
    return rho_to_r(rho, N)


def _haar_noise(N: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure state's Bloch vector. `|s|^2 = d-1` exactly for every draw (purity
    1), not just in expectation -- the cheap, per-sample version of the Q0 check."""
    d = 2 ** N
    out = np.empty((batch, d), dtype=np.complex128)
    for i in range(batch):
        u = haar_random_unitary(N, rng)
        out[i] = u[:, 0]
    rho = np.einsum('bi,bj->bij', out, out.conj())
    return rho_to_r(rho, N)


def _symplectic_codes(labels: list[str]) -> np.ndarray:
    """`(K, N, 2)` binary `(x, z)` symplectic code per Pauli label: I->(0,0), X->(1,0),
    Z->(0,1), Y->(1,1)."""
    table = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    return np.array([[table[c] for c in label] for label in labels], dtype=np.int64)


def _pauli_twirl_noise(r0: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """`s = xi (x) r0`, `xi_k = (-1)^{symplectic_product(P_rand, P_k)}` for one uniformly
    random Pauli string `P_rand` (over all `4^N` strings, identity included) drawn fresh
    per sample. Exact: `sigma = P_rand rho_0 P_rand^dagger` has Pauli vector `xi (x) r0`,
    so this *is* the single-sample realization of `rho_t = abar rho_0 + (1-abar) P rho_0
    P^dagger` -- the twirl of the depolarizing channel (`D_lambda(rho) = sum_k p_k P_k rho
    P_k`, `p_k=(1-lambda)/d^2` for `k!=0`), realized one random Pauli at a time rather than
    averaged in closed form. Preserves purity of `rho_0` exactly (conjugation by a unitary),
    so -- unlike `hs`/`haar` -- this ensemble's terminal prior depends on the data, a
    limitation flagged here and in `docs/QSTATE_DIFFUSION.md`; not used as the default.
    """
    labels, _ = pauli_basis(N)
    codes = _symplectic_codes(labels)  # (K, N, 2)
    batch = r0.shape[0]
    rand_idx = rng.integers(0, 4 ** N, size=batch)
    # Decode each random index into an N-digit base-4 string over 'IXYZ'.
    digits = np.zeros((batch, N), dtype=np.int64)
    tmp = rand_idx.copy()
    for q in range(N - 1, -1, -1):
        digits[:, q] = tmp % 4
        tmp //= 4
    code_map = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)  # I,X,Y,Z order
    rand_codes = code_map[digits]  # (batch, N, 2): (x, z) per qubit
    rand_x, rand_z = rand_codes[..., 0], rand_codes[..., 1]  # (batch, N)
    basis_x, basis_z = codes[..., 0], codes[..., 1]  # (K, N)
    # symplectic product per (sample, basis element k): sum_q (x_rand,q*z_k,q + z_rand,q*x_k,q) mod 2
    sp = np.einsum('bn,kn->bk', rand_x, basis_z) + np.einsum('bn,kn->bk', rand_z, basis_x)
    xi = 1.0 - 2.0 * (sp % 2)
    return xi * r0


def sample_noise(noise_ensemble: str, N: int, batch: int, rng: np.random.Generator,
                 r0: np.ndarray | None = None) -> np.ndarray:
    """`E[s] = 0` Bloch-vector noise from the reference ensemble `E`. `r0` is required
    (and used) only for `noise_ensemble='pauli_twirl'`."""
    if noise_ensemble == 'hs':
        return _hs_noise(N, batch, rng)
    if noise_ensemble == 'haar':
        return _haar_noise(N, batch, rng)
    if noise_ensemble == 'pauli_twirl':
        if r0 is None:
            raise ValueError("pauli_twirl noise needs r0 (it conjugates the data)")
        return _pauli_twirl_noise(r0, N, rng)
    raise ValueError(f"unknown noise_ensemble {noise_ensemble!r}")


def forward_sample(r0: np.ndarray, t: np.ndarray, alpha_bar: np.ndarray, N: int,
                   noise_ensemble: str, rng: np.random.Generator):
    """`r_t = abar_t r0 + (1-abar_t) s`, `s ~ sample_noise(noise_ensemble)`.

    `t` is 1-indexed into `alpha_bar` (`ddpm.cosine_schedule`'s convention). Returns
    `(r_t, s, abar)`.
    """
    abar = alpha_bar[np.asarray(t, dtype=int) - 1]
    s = sample_noise(noise_ensemble, N, len(r0), rng, r0=r0)
    r_t = abar[:, None] * r0 + (1.0 - abar)[:, None] * s
    return r_t, s, abar


def forward_sample_deterministic(r0: np.ndarray, t: np.ndarray, alpha_bar: np.ndarray):
    """QGDM's own deterministic channel (`s == 0` identically): `r_t = abar_t r0`. Used only
    as the Q1 control that shows the stochastic forward removes the closed-form triviality."""
    abar = alpha_bar[np.asarray(t, dtype=int) - 1]
    return abar[:, None] * r0, abar
