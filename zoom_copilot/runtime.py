"""Runtime helpers; no credentials or private document inspection."""
from __future__ import annotations
import os
from pathlib import Path
import site
import sys

_DLL_HANDLES: list = []


def configure_cuda_dlls() -> list[str]:
    """Find optional NVIDIA pip DLLs without modifying the machine PATH."""
    if os.name != "nt":
        return []
    roots = [Path(p) for p in site.getsitepackages()]
    if getattr(sys, "frozen", False):
        roots += [Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))]
    extra = os.environ.get("PRISM_INTERVIEW_CUDA_DIR", "")
    extra_roots = [Path(p) for p in extra.split(os.pathsep) if p]
    paths = list(extra_roots)
    roots += extra_roots
    for root in roots:
        paths.extend(root / "nvidia" / name / "bin" for name in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"))
    found = []
    for path in paths:
        if path.is_dir():
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(path)))
                found.append(str(path))
            except OSError:
                pass
    if found:
        # CTranslate2 loads some CUDA DLLs lazily with legacy LoadLibrary.
        # AddDllDirectory alone does not cover that code path. This changes
        # only this process and its children, never the machine/user PATH.
        current = os.environ.get("PATH", "")
        new = [p for p in found if p not in current.split(os.pathsep)]
        if new:
            os.environ["PATH"] = os.pathsep.join(new + [current])
    return found


def default_model_cache() -> Path:
    override = os.environ.get("PRISM_INTERVIEW_MODELS")
    if override:
        path = Path(override)
    else:
        from .config import app_dir
        path = app_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
