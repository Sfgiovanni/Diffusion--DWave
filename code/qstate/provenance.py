"""Provenance block for every `qstate` result file -- the guardrail is "a result without
provenance is discarded," so every gate script writes this alongside its numbers.
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone

# `git` is not on PATH inside the activated `qrc` conda environment (it lives in the base
# miniconda install's bin/, which `conda activate` does not append) -- found the hard way
# when the first three provenance blocks silently recorded "<unavailable>" for the SHA.
# `shutil.which` first, then these fixed fallbacks, so a working git on this machine is
# actually found instead of masking a real environment gap as a git absence.
_GIT_FALLBACKS = ["/home/franco/miniconda3/bin/git", "/usr/bin/git", "/usr/local/bin/git"]


def _git_executable() -> str:
    found = shutil.which("git")
    if found:
        return found
    for candidate in _GIT_FALLBACKS:
        if shutil.os.path.exists(candidate):
            return candidate
    return "git"  # let it fail with the OS's own error if truly absent


def _run(cmd: list[str], strip: bool = True) -> str:
    cmd = [_git_executable(), *cmd[1:]] if cmd and cmd[0] == "git" else cmd
    try:
        out = subprocess.check_output(cmd, cwd=None, stderr=subprocess.DEVNULL).decode()
        return out.strip() if strip else out.rstrip("\n")
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"<unavailable: {exc}>"


def git_block() -> dict:
    """SHA plus a dirty-tree digest. The working tree in this session had unrelated changes
    from a concurrent D-Wave run (`anneal/`, `experiments/qrc_fusion_fair_generation.py`,
    `run_dwave_qpu_cell.sh`, `results/dwave_qpu_ledger.json`, and its logs) -- a bare SHA
    would misleadingly imply a clean tree, so `dirty_paths` records exactly which paths were
    modified, not touched by this work.

    `status` is read with `strip=False`: `--porcelain`'s fixed-width `XY ` status prefix on
    every line starts with a space for a worktree-only change, and blob-level `.strip()`
    (correct for `rev-parse HEAD`'s single-line output) silently eats exactly that leading
    space on line 1 only, truncating its filename by one character. Caught by noticing
    "nneal/qpu_budget.py" in an actual run rather than assumed absent.
    """
    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"], strip=False)
    dirty_paths = [line[3:] for line in status.splitlines() if line.strip()]
    return dict(sha=sha, dirty_paths=dirty_paths,
               note="dirty_paths from a concurrent, unrelated D-Wave QPU run; none touched by qstate/")


def env_block(device: str = "cuda") -> dict:
    import numpy
    import scipy
    block = dict(numpy=numpy.__version__, scipy=scipy.__version__, device=device)
    try:
        import torch
        block["torch"] = torch.__version__
        block["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            block["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        block["torch"] = "<not installed>"
    return block


def provenance(script: str, seeds: dict, hyperparameters: dict, device: str = "cuda") -> dict:
    return dict(
        script=script,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git=git_block(),
        env=env_block(device),
        seeds=seeds,
        hyperparameters=hyperparameters,
    )
