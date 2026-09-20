# Diffusion--DWave

Appendix repository for the quantum-state-diffusion paper: a fixed quantum reservoir (QRC)
replacing the trainable denoiser in a QGDM-style diffusion process over `N`-qubit density
matrices, compared against a classical ridge baseline, on three target-state families
(TFIM ground states, QuDDPM's "circular states", QuDDPM's "correlated noise"), on three
substrates: a **digital reservoir** (local GPU simulator), a **real D-Wave QPU**
(`Advantage_system4`), and the free **`dwave-sqa`** software annealing simulator.

## Layout

- **`paper_figures/`** -- the three main figures (physical structure of each target family
  overlaid with real-QPU-vs-`dwave-sqa` reconstruction fidelity), each with its PNG/PDF and
  two CSVs (`fidelity_by_bin.csv`: mean +/- SEM per bin per arm; `fidelity_by_seed.csv`: the
  raw per-seed values behind it).
- **`digital_appendix/`** -- the digital-reservoir arm's data for the same three families
  (appendix material): `tfim_phase_diagram_digital/` (g-resolved, same bin convention as
  `paper_figures/tfim_phase_diagram/`, plus its own figure) and `circular_digital/` /
  `correlated_noise_digital/` (aggregate fidelity per seed x unitary draw -- these two
  digital runs were not g/x/delta-resolved, unlike their real-QPU/`dwave-sqa` counterparts
  in `paper_figures/`).
- **`code/`** -- the scripts that produced the data above: the `qstate/` package (target
  generation, the stochastic forward process, Pauli-vector algebra, the ridge-readout arms),
  its `anneal/`/`quera/`/`magicqrc/` dependencies (D-Wave feature extraction and budget
  tracking, the digital reservoir's fixed-unitary mechanics), and `experiments/`/`figures/`
  (the per-family scripts and plotting code). See `code/README.md`.

## Method, briefly

Each target family is a distribution over `N`-qubit density matrices (as a Bloch/Pauli
vector `r0`). A stochastic forward process `r_t = abar_t * r0 + (1 - abar_t) * s`
(`s` zero-mean noise) plays the role of the diffusion forward pass; the task is to recover
`r0` from `(r_t, t)`. Three readouts are compared, all a ridge regression on top of a
feature map:

- **classical**: features are `[r_t | sinusoidal-time-embedding]` directly (no reservoir).
- **QRC (digital reservoir)**: `r_t` is encoded into a small fixed digital quantum circuit
  (untrained, fixed unitary draws) and its measurement statistics become extra features.
- **QRC (real D-Wave QPU / `dwave-sqa`)**: `r_t` is encoded into local fields of a fixed
  Ising problem, annealed on real hardware or D-Wave's own software simulator, and the
  resulting spin statistics become extra features.

Only the ridge readout weights are ever trained; the reservoir/annealer dynamics are fixed
and untrained in every case. Full narrative, every ablation, and every caveat live in the
parent research repository's `docs/QSTATE_DIFFUSION.md` (not included here).
