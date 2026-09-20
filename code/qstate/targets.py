"""Target-state generation: random pure states and probabilistic mixtures of two, per the
protocol this task specifies for QGDM (arXiv:2401.07039).

**Adaptation, not verbatim reproduction.** The paper's numerical section (VI-A/VI-B) states
random pure targets are "random-circuit-generated" and random mixed targets are "low-rank
mixtures of random-circuit-generated pure states," but defers the exact circuit ansatz,
gate set and layer count to a Supplementary File not available to this session (checked via
`arxiv.org/abs/2401.07039` and its HTML rendering, 2026-09-15 -- both give the equations
and the sentence above but not the circuit itself). The mixture construction (softmax of
`U(0,1]` weights over two pure states) IS fully specified by this task's prompt and is
followed exactly.

The ansatz used here: `depth` layers of `(RY(theta), RZ(phi))` per qubit with
`theta, phi ~ U(0, pi)` (matching the prompt's stated parameter range), each layer
followed by a ring of CNOTs for `N > 1`, applied to `|0...0>`. This is a standard
hardware-efficient ansatz and a reasonable stand-in, not a claim of matching the paper's
exact circuit -- flagged again in `docs/QSTATE_DIFFUSION.md`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magicqrc.circuits import embed_1q, embed_cnot
from qstate.pauli import rho_to_r

DEFAULT_DEPTH = 3


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(phi: float) -> np.ndarray:
    return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=np.complex128)


def random_statevector(N: int, rng: np.random.Generator, depth: int = DEFAULT_DEPTH) -> np.ndarray:
    """Statevector `|psi> = U(theta) |0...0>` for the layered random ansatz above.

    Gates are applied as full `(d, d)` unitaries (`embed_1q`/`embed_cnot`, reused from
    `magicqrc.circuits`) rather than tensor-axis tricks -- `d <= 8` at the qubit budget this
    object works at (`docs/QSTATE_DIFFUSION.md`), so the O(d^2) matvec cost is negligible and
    not worth risking an axis-order bug for.
    """
    d = 2 ** N
    psi = np.zeros(d, dtype=np.complex128)
    psi[0] = 1.0
    for _ in range(depth):
        for q in range(N):
            theta = rng.uniform(0.0, np.pi)
            phi = rng.uniform(0.0, np.pi)
            psi = embed_1q(_ry(theta), q, N) @ psi
            psi = embed_1q(_rz(phi), q, N) @ psi
        if N > 1:
            for q in range(N):
                psi = embed_cnot(q, (q + 1) % N, N) @ psi
    return psi


def random_pure_r(N: int, rng: np.random.Generator, depth: int = DEFAULT_DEPTH) -> np.ndarray:
    """`(4^N-1,)` Pauli vector of one random pure target state."""
    psi = random_statevector(N, rng, depth)
    rho = np.outer(psi, psi.conj())[None]
    return rho_to_r(rho, N)[0]


def random_mixed_r(N: int, rng: np.random.Generator, depth: int = DEFAULT_DEPTH):
    """Softmax(`U(0,1]`, `U(0,1]`)-weighted mixture of two random pure states.

    Returns `(r0, weights)`. `E[rho] != I/d` in general (this is a *target* ensemble, not a
    noise reference -- do not conflate with `forward.sample_noise('hs', ...)`, which is a
    different construction entirely).
    """
    psi1 = random_statevector(N, rng, depth)
    psi2 = random_statevector(N, rng, depth)
    u = 1.0 - rng.random(2)  # (0, 1], per the prompt's protocol
    weights = np.exp(u) / np.exp(u).sum()
    rho1 = np.outer(psi1, psi1.conj())
    rho2 = np.outer(psi2, psi2.conj())
    rho = weights[0] * rho1 + weights[1] * rho2
    r0 = rho_to_r(rho[None], N)[0]
    return r0, weights


@dataclass
class TargetPool:
    """A fixed pool of target Pauli vectors plus their generation metadata."""
    r0: np.ndarray          # (n, 4^N-1)
    kind: str                # 'pure' or 'mixed'
    N: int
    seed: int
    depth: int


def generate_pool(N: int, n: int, seed: int, kind: str = 'mixed',
                  depth: int = DEFAULT_DEPTH) -> TargetPool:
    """`n` independent draws from `random_pure_r`/`random_mixed_r`, seeded by `seed`."""
    rng = np.random.default_rng(seed)
    if kind == 'pure':
        r0 = np.stack([random_pure_r(N, rng, depth) for _ in range(n)])
    elif kind == 'mixed':
        r0 = np.stack([random_mixed_r(N, rng, depth)[0] for _ in range(n)])
    else:
        raise ValueError(f"unknown target kind {kind!r}")
    return TargetPool(r0=r0, kind=kind, N=N, seed=seed, depth=depth)


def generate_clustered_pool(n: int, seed: int, epsilon: float = 0.08) -> TargetPool:
    """QuDDPM's (arXiv:2310.05866) Table I `N=1` "clustered state" benchmark target family:
    `|psi> = (|0> + eps*c1*|1>) / norm`, `c1 ~ CN(0,1)` (standard complex Gaussian), `eps` the
    paper's own value (0.08, Appendix A). `N=1` only -- this is specifically their
    single-qubit benchmark; the paper does not give a general-`N` construction for it.

    Added to test a concrete question: does this project's method (fixed reservoir + linear
    readout, still under THIS project's own stochastic forward process, not QuDDPM's
    scrambling one) also saturate near QuDDPM's reported ~0.99 fidelity on their own easy
    (narrow-cluster) target family and evaluation convention (fidelity to the cluster
    CENTER `|0>`, not to each sample's own true origin state -- see
    `qstate/metrics.fidelity_to_reference`)? This isolates "is the target distribution's
    narrowness what explains the gap" from every other difference between the two projects
    (forward process, training regime, metric) -- those other differences are NOT removed
    here, only this one, and `docs/QSTATE_DIFFUSION.md` states that plainly rather than
    implying a full replication.
    """
    rng = np.random.default_rng(seed)
    c1 = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = np.stack([np.ones(n, dtype=np.complex128), epsilon * c1], axis=1)
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    rho = np.einsum('ni,nj->nij', psi, psi.conj())
    r0 = rho_to_r(rho, 1)
    return TargetPool(r0=r0, kind=f'clustered_eps{epsilon:g}', N=1, seed=seed, depth=0)


def tfim_hamiltonian(N: int, J: float, g: float) -> np.ndarray:
    """`H = -J sum_i Z_i Z_{i+1} - g sum_i X_i`, open boundary, `N` qubits. At `N=1` the `ZZ`
    sum is empty (`H = -g X`), whose ground state (`|->`) does not depend on `g`'s magnitude
    for any `g>0` -- `generate_tfim_pool` documents this degeneracy rather than silently
    returning a constant "family"."""
    d = 2 ** N
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    H = np.zeros((d, d), dtype=np.complex128)
    for i in range(N - 1):
        H -= J * embed_1q(Z, i, N) @ embed_1q(Z, i + 1, N)
    for i in range(N):
        H -= g * embed_1q(X, i, N)
    return H


def tfim_ground_state_r(N: int, J: float, g: float) -> np.ndarray:
    H = tfim_hamiltonian(N, J, g)
    w, v = np.linalg.eigh(H)
    psi = v[:, 0]
    rho = np.outer(psi, psi.conj())[None]
    return rho_to_r(rho, N)[0]


def generate_tfim_pool(N: int, n: int, seed: int, J: float = 1.0,
                       g_range: tuple[float, float] = (0.0, 2.0)):
    """A 1-parameter (`g`, the transverse field) family of TFIM ground states -- a
    physically-structured, low-dimensional (a smooth curve in the `4^N-1`-dim Bloch space,
    not an isotropic fill of it) target distribution, unlike `generate_pool`'s Haar-adjacent
    one. Added to test whether the reservoir shows an advantage on targets that have actual
    structure to exploit, per the `N=1`-vs-`N>=2` and "is Haar-random the worst possible
    benchmark for a fixed physical reservoir" discussion in `docs/QSTATE_DIFFUSION.md`.
    `N=1` is degenerate here (ground state of `-g X` is `|->` for every `g>0`, see
    `tfim_hamiltonian`'s docstring) -- use `N>=2`. Returns `(TargetPool, g_values)`.
    """
    rng = np.random.default_rng(seed)
    g_vals = rng.uniform(g_range[0], g_range[1], size=n)
    r0 = np.stack([tfim_ground_state_r(N, J, float(g)) for g in g_vals])
    pool = TargetPool(r0=r0, kind=f'tfim_J{J:g}_g{g_range[0]:g}-{g_range[1]:g}', N=N,
                      seed=seed, depth=0)
    return pool, g_vals


def generate_circular_pool(n: int, seed: int) -> tuple[TargetPool, np.ndarray]:
    """QuDDPM's (arXiv:2310.05866) "circular states" / topology benchmark (Fig. 6, Table I
    row "Circular states"): `|psi_i> = e^{-i*x_i*G}|0>`, single qubit, generator `G = Y`
    (Pauli Y), `x_i ~ U[0, 2*pi]`. Closed form (Y|0>=i|1>, `Y^2=I`, so
    `e^{-ixY}|0> = cos(x)|0> - i*sin(x)*Y|0> = cos(x)|0> + sin(x)|1>`, real amplitudes,
    `<Y>=0` for every state -- matches the paper's own framing of this family living in the
    Bloch sphere's X-Z plane): a 1-parameter *ring* (not a line segment/curve endpoint, `x`
    wraps at `2*pi`) of pure states, the paper's own "nontrivial topology" example.

    Different in kind from `generate_tfim_pool`'s curve: no Hamiltonian, no physical
    parameter with a phase transition -- a purely geometric family, and the cheapest one
    this project can test (`N=1`, single qubit, fits every arm including real QPU/Rydberg
    budgets). Their own evaluation metric (`<Y>^2` deviation from the unit circle via
    Wasserstein distance, Fig. 6(d)-(e)) is NOT reproduced here -- as with
    `generate_clustered_pool`, this project's own stochastic forward and fidelity-based
    metric are used instead (see `docs/QSTATE_DIFFUSION.md`); only the target *distribution*
    is matched. Returns `(TargetPool, x_values)`.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2 * np.pi, size=n)
    psi = np.stack([np.cos(x), np.sin(x)], axis=1).astype(np.complex128)
    rho = np.einsum('ni,nj->nij', psi, psi.conj())
    r0 = rho_to_r(rho, 1)
    pool = TargetPool(r0=r0, kind='circular', N=1, seed=seed, depth=0)
    return pool, x


def generate_correlated_noise_pool(n: int, seed: int, delta0: float = np.pi / 3,
                                   p: float = 0.5,
                                   c: tuple[complex, complex, complex, complex] | None = None
                                   ) -> tuple[TargetPool, np.ndarray, np.ndarray]:
    """QuDDPM's (arXiv:2310.05866) "correlated noise" benchmark (Fig. 5(a), Table I row
    "Correlated noise"): a FIXED 2-qubit clean target `|Psi> = c0|00> + c1|01> + c3|11>`
    (their own notation -- the `|10>` coefficient is pinned to 0, not a typo) under
    correlated coherent gate noise: with probability `p` apply `e^{-i*delta*X1X2}`,
    otherwise `e^{-i*delta*Z1Z2}` (both closed-form since `(X1X2)^2=(Z1Z2)^2=I`), angle
    `delta ~ U[-delta0, delta0]`. `delta0 = pi/3` is the paper's own value (Appendix A).

    **Two parameters the paper's main text does not give a specific numeric value for in
    the pages read for this addition** (only the family's functional form and `delta0`):
    the exact `(c0,c1,c3)` amplitudes and the noise-type probability `p`. Defaults here --
    `c=(1,1,0,1)/sqrt(3)` (an equal-weight superposition over the three populated basis
    states, a natural reading of writing all three coefficients without further
    qualification) and `p=0.5` (an unbiased coin between the two noise channels, the
    simplest reading of "happen with probability p and 1-p" absent a stated value) -- are
    stated explicitly as this project's own choice, not the paper's, per this document's
    standing practice for underspecified details (see `qstate/targets.py`'s module
    docstring on the QGDM ansatz for the same pattern).

    Models a physically-motivated "device characterization" target family: unlike
    `generate_tfim_pool`'s Hamiltonian-ground-state curve or `generate_circular_pool`'s pure
    geometry, here the *noise itself* (not this project's own forward-diffusion process) is
    what scatters the ensemble around one fixed clean state -- a genuinely different kind of
    structure than anything else this document has tested. Output states stay pure (unitary
    noise). Returns `(TargetPool, delta_values, used_xx_mask)`.
    """
    if c is None:
        c = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    c = c / np.linalg.norm(c)
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-delta0, delta0, size=n)
    use_xx = rng.random(n) < p
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    XX = np.kron(X, X)
    ZZ = np.kron(Z, Z)
    I4 = np.eye(4, dtype=np.complex128)
    G = np.where(use_xx[:, None, None], XX[None], ZZ[None])
    cosd = np.cos(delta)[:, None, None]
    sind = np.sin(delta)[:, None, None]
    U = cosd * I4[None] - 1j * sind * G
    psi = np.einsum('nij,j->ni', U, c)
    rho = np.einsum('ni,nj->nij', psi, psi.conj())
    r0 = rho_to_r(rho, 2)
    pool = TargetPool(r0=r0, kind=f'correlated_noise_p{p:g}_d{delta0:g}', N=2, seed=seed, depth=0)
    return pool, delta, use_xx
