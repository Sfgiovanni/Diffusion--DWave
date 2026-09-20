# Code

A minimal, working excerpt of the parent research repository -- only what's needed to
reproduce the data in `../paper_figures/` and `../digital_appendix/`.

## Layout

```
qstate/            target generation, stochastic forward process, Pauli algebra, ridge arms
anneal/            D-Wave feature extraction (real QPU + dwave-sqa), QPU budget tracking
quera/             affine encoding fit + a small on-disk cache
magicqrc/          fixed-unitary circuit families (Haar, Clifford, alpha-dial, chaotic Ising)
denoiser_qrc.py    the digital reservoir's fixed quantum circuit mechanics
denoiser_classical.py   sinusoidal time embedding
ddpm.py            cosine noise schedule (`abar_t`)
experiments/       one script per (family, substrate) -- see below
figures/           one plotting script per figure in ../paper_figures/ and
                   ../digital_appendix/tfim_phase_diagram_digital/
```

## `experiments/` scripts, by family and substrate

| family | digital | real QPU | `dwave-sqa` |
|---|---|---|---|
| TFIM (N=2), g-resolved | `qstate_gate2_tfim_phase_diagram_digital.py` | `qstate_gate2_tfim_phase_diagram_qpu.py` | `qstate_gate2_tfim_phase_diagram_sqa.py` |
| circular states (N=1) | `qstate_gate2_circular_digital.py` (aggregate) | `qstate_gate2_circular_diagram_qpu.py` (x-resolved) | `qstate_gate2_circular_diagram_sqa.py` (x-resolved) |
| correlated noise (N=2) | `qstate_gate2_correlated_noise_digital.py` (aggregate) | `qstate_gate2_correlated_noise_diagram_qpu.py` (delta-resolved) | `qstate_gate2_correlated_noise_diagram_sqa.py` (delta-resolved) |

Each script writes a parquet of per-cell (or per-bin) fidelity; the CSVs in
`../paper_figures/` and `../digital_appendix/` are that data, aggregated and exported (see
each folder's own files -- `fidelity_by_bin.csv`/`fidelity_by_seed.csv` or
`fidelity_summary.csv`/`fidelity_by_seed_draw.csv`).

Real-QPU scripts (`*_qpu.py`) require D-Wave Ocean SDK credentials and spend real,
non-refundable QPU time (tracked against a small hard-capped budget,
`anneal.qpu_budget.QpuBudget`) -- they are included for transparency/reproducibility of the
method, not meant to be re-run casually. `dwave-sqa` scripts need only
`dwave-samplers` (a local, free, CPU-only annealing simulator). Digital scripts need a
CUDA GPU (`denoiser_qrc.py`'s reservoir simulation) but no external service.

This excerpt keeps only the classical / QRC-digital / real-QPU / `dwave-sqa` comparison.
