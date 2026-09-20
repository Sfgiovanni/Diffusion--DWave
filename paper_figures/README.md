# State-space diagram figures

Three figures, each showing a target-state family's own physical structure (top panel) overlaid with reconstruction fidelity of the classical baseline, real D-Wave QPU, and the free `dwave-sqa` simulator (bottom panel), at matched settings (n_train=n_val=100, n_test=200, 3 seeds, num_reads=20) so QPU and simulator are a genuine apples-to-apples overlay.

Each subfolder has:
- `<name>.png` / `.pdf` -- the figure
- `fidelity_by_bin.csv` -- mean +/- SEM per bin per arm (classical/qpu/sqa), what the figure plots
- `fidelity_by_seed.csv` -- the raw per-seed, per-bin fidelity values behind that aggregate

## tfim_phase_diagram
TFIM ground state (N=2), transverse field `g` in [0,2] (J=1), 10 bins.

## circular_diagram
QuDDPM 'circular states' (N=1), Bloch angle `x` in [0,2*pi), 12 bins. Purely geometric family, no phase transition.

## correlated_noise_diagram
QuDDPM 'correlated noise' (N=2), error angle `delta` in [-pi/3,pi/3], 10 bins, both noise channels (X1X2/Z1Z2) pooled.
