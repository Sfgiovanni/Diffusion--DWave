"""Second generative object: N-qubit quantum states via a QGDM-style diffusion whose
denoiser is the fixed QRC reservoir (`denoiser_qrc.py`), not a trained PQC.

See `docs/QSTATE_DIFFUSION.md` for the design rationale, the two degeneracies that shape
it (deterministic-forward triviality, Haar-target isotropy), and gate-by-gate results.
"""
