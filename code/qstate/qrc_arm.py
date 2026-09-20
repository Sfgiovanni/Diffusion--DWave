"""Digital QRC feature extraction for Pauli-vector inputs, reusing `denoiser_qrc.py`
unchanged (per the task's no-semantic-change rule -- this module only calls its public
functions with a different input vector, it edits nothing in it).

The Pauli vector `r_t` plays the role the PCA latent `x_t` plays in the image pipeline: it
is reset into the reservoir's data qubits every step, exactly like `qrc_fusion_fair_core
.qrc_features` does for `x_t`. Two things are specific to this object and handled here:

1. **Odd channel width.** `denoiser_qrc._input_density` reshapes the input into
   `(n_data_qubits, 2)` pairs, so an odd number of channels crashes. `N=1` gives 3 Pauli
   components (+2 if `inject_time_channel`) = 5, odd; `N=2` gives 15(+2) = 17, odd. Both are
   zero-padded up to the next even width (one dummy channel) rather than changing the
   encoding -- the extra channel carries no information (it is always 0) and costs one
   reset qubit's worth of denominator in the tensor product, not a behavior change.
2. **Reservoir sizing.** Reset qubits = padded_width // 2 (3 for N=1, 9 for N=2 with time
   injection on). Memory qubits fill the remaining budget up to the `denoiser_qrc.py`
   docstring's documented n=9 practical ceiling (3 for N=1, 0 for N=2 -- N=2 is stateless
   at this qubit budget, noted in `docs/QSTATE_DIFFUSION.md` rather than silently implied
   to behave like N=1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from denoiser_qrc import data_qubits, fixed_slices, initial_state, inject_time, step

QSTATE_ROOT = 20260915
PRACTICAL_QUBIT_CEILING = 9
DEFAULT_MEMORY_CAP = 3
DEFAULT_V_SLICES = 4


@dataclass
class ReservoirConfig:
    n_reset: int
    n_memory: int
    n_qubits: int
    padded_width: int
    n_pad: int


def reservoir_layout(raw_width: int, memory_cap: int = DEFAULT_MEMORY_CAP,
                     ceiling: int = PRACTICAL_QUBIT_CEILING) -> ReservoirConfig:
    n_pad = raw_width % 2
    padded_width = raw_width + n_pad
    n_reset = padded_width // 2
    n_memory = max(0, min(memory_cap, ceiling - n_reset))
    return ReservoirConfig(n_reset=n_reset, n_memory=n_memory, n_qubits=n_reset + n_memory,
                           padded_width=padded_width, n_pad=n_pad)


@torch.inference_mode()
def qrc_pauli_features(r_t: np.ndarray, t: np.ndarray, T: int, draw: int, device: str,
                       encoding: str = 'quadrature', v_slices: int = DEFAULT_V_SLICES,
                       inject_time_channel: bool = True, t_scale: float = 1.0,
                       memory_cap: int = DEFAULT_MEMORY_CAP, batch: int = 2048,
                       correlations: bool = True, observables: str = 'zz'):
    """`(n, K)` Pauli vector `r_t` -> `(n, n_features)` reservoir features.

    One fresh reservoir preparation (`initial_state` + `reset_data` inside `step`) per row,
    `v_slices` sequential unitary applications read out after each -- byte-for-byte the same
    protocol `qrc_fusion_fair_core.qrc_features` uses for the image latent, with `r_t` (plus
    an optional (sin,cos) time channel from the unmodified `denoiser_qrc.inject_time`) as the
    reset-qubit input instead of the PCA latent.

    `observables='zz'` (default) reads `<Z_q>`/`<Z_q Z_r>` off the post-unitary diagonal,
    via `denoiser_qrc.step` -- unchanged from before this parameter existed.
    `observables='full'` instead reads the full weight-<=2 Pauli family (`<X_q>`, `<Y_q>`,
    `<Z_q>`, and all `XX`/`XY`/.../`ZZ` pairs) via `qrc_fusion_fair_core.step_full`/
    `pauli_ops`, reused unmodified. Motivated by a concrete asymmetry between the two input
    encodings: `quadrature` keeps a live Z-component in its input density
    (`z=sqrt(1-a^2-b^2)`), directly visible to a Z/ZZ-only readout, while `multibase` pins
    `z=0` always, so ALL of its input information depends on the fixed reservoir unitary
    rotating X/Y into Z before a Z/ZZ readout can see it -- `observables='full'` removes that
    asymmetry by reading X/Y/Z (and their pairwise correlators) directly, regardless of how
    much the fixed dynamics happens to rotate. Feature count grows sharply (`3n + 9*C(n,2)`
    vs. `n + C(n,2)` per slice -- e.g. 153 vs. 21 per slice at `n=6`), so this needs more
    training data or a wider `LAMBDAS` search to stay well-conditioned; not a free upgrade.
    """
    raw_width = r_t.shape[1] + (2 if inject_time_channel else 0)
    layout = reservoir_layout(raw_width, memory_cap)
    slices = fixed_slices(v_slices, QSTATE_ROOT + 500 + draw, device, layout.n_qubits,
                          np.pi / 4, 4, 'alpha_dial')
    ops = None
    if observables == 'full':
        from experiments.qrc_fusion_fair_core import pauli_ops
        ops = pauli_ops(layout.n_qubits, device)
    out = []
    for i in range(0, len(r_t), batch):
        xb = torch.as_tensor(r_t[i:i + batch], dtype=torch.float32, device=device)
        tb = torch.as_tensor(t[i:i + batch], device=device)
        if inject_time_channel:
            xb = inject_time(xb, tb, T, t_scale)
        if layout.n_pad:
            pad = torch.zeros((xb.shape[0], layout.n_pad), dtype=xb.dtype, device=device)
            xb = torch.cat([xb, pad], dim=1)
        assert xb.shape[1] == layout.padded_width
        state = initial_state(len(xb), device, layout.n_qubits)
        assert data_qubits(xb, encoding) == layout.n_reset
        if observables == 'full':
            from experiments.qrc_fusion_fair_core import step_full
            _, feats = step_full(state, xb, slices, encoding, None, ops)
        else:
            _, feats = step(state, xb, slices, encoding, correlations, 1.0, time_phase=None)
        out.append(feats.cpu().numpy())
    return np.concatenate(out), layout
