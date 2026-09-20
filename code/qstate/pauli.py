"""Generalized Pauli-vector algebra for N-qubit density matrices.

`rho <-> r` is the same linear parametrization the forward/backward diffusion is defined
in: `r_P = tr(rho P)` for `P` ranging over the `4^N - 1` non-identity N-fold Pauli strings,
and `rho = (I + sum_P r_P P) / d`. Every function here is batched over a leading axis and
works for any `N`; the practical qubit budget (`N in {1,2}`, see `docs/QSTATE_DIFFUSION.md`)
is enforced by the callers, not by this module.

Ordering convention: `pauli_labels(N)` enumerates `itertools.product('IXYZ', repeat=N)`
in that fixed lexicographic order, dropping the all-identity string. Every array indexed by
"Pauli component" in this codebase uses that order; nothing here reorders it later.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

_PAULI = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_labels(N: int) -> list[str]:
    identity = 'I' * N
    return [''.join(p) for p in itertools.product('IXYZ', repeat=N) if ''.join(p) != identity]


def _pauli_matrix(label: str) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for c in label:
        m = np.kron(m, _PAULI[c])
    return m


@lru_cache(maxsize=None)
def pauli_basis(N: int):
    """`(labels, mats)`: `4^N - 1` labels and their `(d, d)` matrices, `d = 2^N`.

    Cached (small, `N in {1,2,3}` in practice) so repeated calls in a rollout loop don't
    rebuild the tensor-product basis every step.
    """
    labels = pauli_labels(N)
    mats = np.stack([_pauli_matrix(l) for l in labels])
    return labels, mats


def n_qubits_from_dim(K: int) -> int:
    """Invert `K = 4^N - 1` for the component count of a Pauli vector."""
    N = round(np.log(K + 1) / np.log(4))
    if 4 ** N - 1 != K:
        raise ValueError(f"{K} is not 4^N - 1 for any integer N")
    return N


def rho_to_r(rho: np.ndarray, N: int | None = None) -> np.ndarray:
    """`(batch, d, d)` complex -> `(batch, 4^N-1)` real Pauli vector."""
    d = rho.shape[-1]
    if N is None:
        N = round(np.log2(d))
    _, mats = pauli_basis(N)
    # tr(rho @ P) is real for Hermitian rho, P; the .real drop is exact up to fp noise.
    return np.einsum('bij,kji->bk', rho, mats).real.astype(np.float64)


def r_to_rho(r: np.ndarray, N: int | None = None) -> np.ndarray:
    """`(batch, 4^N-1)` real Pauli vector -> `(batch, d, d)` complex density matrix.

    Always trace 1 by construction (the `I/d` term); PSD is NOT guaranteed -- callers that
    need a physical state must go through `psd_project`.
    """
    if N is None:
        N = n_qubits_from_dim(r.shape[-1])
    d = 2 ** N
    _, mats = pauli_basis(N)
    eye = np.eye(d, dtype=np.complex128)
    return (eye[None] + np.einsum('bk,kij->bij', r.astype(np.complex128), mats)) / d


def purity(rho: np.ndarray) -> np.ndarray:
    """`tr(rho^2)`, computed directly (not via the `(1+|r|^2)/d` shortcut) so a bug in the
    latter shows up as a test failure rather than being self-consistently invisible."""
    return np.einsum('bij,bji->b', rho, rho).real


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Uhlmann fidelity `F(rho,sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2 in [0,1]`.

    Both inputs must already be PSD, trace-1 (run `psd_project` first if they came from a
    readout prediction). Eigendecomposition-based, batched; `N=1` cross-checks this against
    the closed form `tr(rho sigma) + 2*sqrt(det rho * det sigma)` in the test suite.
    """
    wr, vr = np.linalg.eigh(rho)
    wr = np.clip(wr.real, 0, None)
    sqrt_rho = (vr * np.sqrt(wr)[:, None, :]) @ np.conj(np.transpose(vr, (0, 2, 1)))
    m = sqrt_rho @ sigma @ sqrt_rho
    m = (m + np.conj(np.transpose(m, (0, 2, 1)))) / 2
    wm = np.clip(np.linalg.eigvalsh(m).real, 0, None)
    return np.sum(np.sqrt(wm), axis=-1) ** 2


def fidelity_qubit_closed_form(rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """`N=1` only: `F = tr(rho sigma) + 2*sqrt(det(rho) det(sigma))`, a cross-check for
    `fidelity`'s eigendecomposition path, not a replacement for it (does not generalize
    past `d=2`)."""
    assert rho.shape[-1] == 2, "closed form only holds for single-qubit (d=2) states"
    tr_prod = np.einsum('bij,bji->b', rho, sigma).real
    det_r = np.linalg.det(rho).real
    det_s = np.linalg.det(sigma).real
    return tr_prod + 2 * np.sqrt(np.clip(det_r, 0, None) * np.clip(det_s, 0, None))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """`0.5 * sum_i |eig_i(rho - sigma)|`."""
    diff = rho - sigma
    diff = (diff + np.conj(np.transpose(diff, (0, 2, 1)))) / 2
    w = np.linalg.eigvalsh(diff).real
    return 0.5 * np.sum(np.abs(w), axis=-1)


def _simplex_project(w: np.ndarray):
    """Euclidean projection of each row of `w` onto the probability simplex
    (Duchi et al. 2008 / equivalently Michelot's algorithm; `w` need not sum to 1).

    Returns `(p, nu)`: the projected point and `nu = sum_i max(0, -w_i)`, the raw negative
    eigenvalue mass *before* projection (the diagnostic `docs/QSTATE_DIFFUSION.md` reports
    per arm per t -- not the mass the projection algorithm mechanically clips, which can
    differ once the positive entries are also shrunk to keep the sum at 1).
    """
    d = w.shape[-1]
    nu = np.sum(np.clip(-w, 0, None), axis=-1)
    u = np.sort(w, axis=-1)[..., ::-1]
    css = np.cumsum(u, axis=-1)
    idx = np.arange(1, d + 1)
    cond = u - (css - 1) / idx > 0
    rho_idx = np.sum(cond, axis=-1)
    theta = (np.take_along_axis(css, (rho_idx - 1)[..., None], axis=-1)[..., 0] - 1) / rho_idx
    p = np.clip(w - theta[..., None], 0, None)
    return p, nu


def psd_project(rho: np.ndarray):
    """Project each `rho` onto {PSD, trace 1} by simplex-projecting its eigenvalues.

    Trace is preserved exactly (simplex projection conserves the sum, and `rho` from
    `r_to_rho` already has trace 1 by construction). Returns `(rho_projected, nu)`.
    """
    rho_h = (rho + np.conj(np.transpose(rho, (0, 2, 1)))) / 2
    w, v = np.linalg.eigh(rho_h)
    w = w.real
    p, nu = _simplex_project(w)
    rho_p = (v * p[:, None, :]) @ np.conj(np.transpose(v, (0, 2, 1)))
    return rho_p, nu
