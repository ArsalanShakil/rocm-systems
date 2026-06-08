# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for Triton.

Wraps the Triton kernel-launch entry points (``JITFunction.run`` and
``CompiledKernel.run`` / ``CompiledKernel.__call__``) so that Triton and
Inductor kernel launches appear in ROCTX markers.
"""

import importlib.util
import threading
from functools import wraps
from pathlib import Path
from typing import Any

from utils.inject_roctx import _core
from utils.inject_roctx._core import (
    _pop_scope,
    _push_scope,
    resolve_user_caller_location,
)
from utils.logger import console_log, console_warning

from . import register

_BACKEND_NAME = "triton"

CompiledKernel: Any = None
JITFunction: Any = None

# Per-thread flag set while a launch marker is open, so nested launch calls
# emit a single marker.
_thread_local = threading.local()


def _in_launch() -> bool:
    return getattr(_thread_local, "in_launch", False)


def _resolve_triton() -> bool:
    """Bind the triton handles. Returns True if triton is importable."""
    global CompiledKernel, JITFunction
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        from triton.compiler import CompiledKernel as _CK

        CompiledKernel = _CK
    except Exception:
        CompiledKernel = None
    try:
        from triton.runtime.jit import JITFunction as _JIT

        JITFunction = _JIT
    except Exception:
        JITFunction = None
    return CompiledKernel is not None or JITFunction is not None


def _register_framework_root() -> None:
    """Register triton's package directory so caller-location resolution
    reports the user's call site rather than triton's internals."""
    try:
        import triton

        triton_file = getattr(triton, "__file__", None)
        if triton_file:
            _core.add_framework_root(str(Path(triton_file).parent))
    except Exception as exc:
        console_warning(
            "api trace",
            f"Could not register triton framework root: {exc}",
        )


def _extract_kernel_name(obj: object, default: str = "<triton_kernel>") -> str:
    """Resolve the kernel name from ``name``, ``metadata``, or ``fn``,
    returning ``default`` when none is available."""
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        meta_name = metadata.get("name")
        if isinstance(meta_name, str) and meta_name:
            return meta_name
    else:
        meta_name = getattr(metadata, "name", None)
        if isinstance(meta_name, str) and meta_name:
            return meta_name

    fn = getattr(obj, "fn", None)
    fn_name = getattr(fn, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name

    return default


def _wrap_launch(
    owner: type,
    method_name: str,
    marker_prefix: str,
) -> bool:
    """Wrap ``owner.method_name`` with a ROCTX range. Idempotent.

    Returns True when the wrapper is installed or already present.
    """
    original = getattr(owner, method_name, None)
    if original is None:
        return False
    if getattr(original, "_roctx_wrapped", False):
        return True

    @wraps(original)
    def launch_with_roctx(self: object, *args: Any, **kwargs: Any) -> object:
        if _in_launch():
            return original(self, *args, **kwargs)
        kernel_name = _extract_kernel_name(self)
        location = resolve_user_caller_location()
        marker = f"{marker_prefix}.{kernel_name}"
        _thread_local.in_launch = True
        _push_scope(marker, f"#1@{location}", backend=_BACKEND_NAME)
        try:
            return original(self, *args, **kwargs)
        finally:
            _pop_scope()
            _thread_local.in_launch = False

    launch_with_roctx._roctx_wrapped = True
    try:
        setattr(owner, method_name, launch_with_roctx)
        console_log(
            "api trace",
            f"Wrapped {owner.__name__}.{method_name} with ROCTX markers",
        )
        return True
    except Exception as exc:
        console_warning(
            "api trace",
            f"Could not patch {owner.__name__}.{method_name}: {exc}",
        )
        return False


def patch_triton_launcher() -> None:
    """Wrap every available Triton launch entry point."""
    wrapped_any = False
    if JITFunction is not None:
        wrapped_any |= _wrap_launch(JITFunction, "run", "triton.JITFunction")
    if CompiledKernel is not None:
        # Prefer run(); fall back to __call__.
        if hasattr(CompiledKernel, "run"):
            wrapped_any |= _wrap_launch(CompiledKernel, "run", "triton.CompiledKernel")
        else:
            wrapped_any |= _wrap_launch(
                CompiledKernel, "__call__", "triton.CompiledKernel"
            )
    if not wrapped_any:
        console_warning(
            "api trace",
            "No Triton launch entry points found to instrument; "
            "Triton API tracing may have no effect.",
        )


class TritonBackend:
    name = "triton"

    def install(self) -> None:
        if not _resolve_triton():
            console_warning(
                "api trace",
                "Triton is not installed; skipping triton instrumentation.",
            )
            return
        if not _core.ensure_python_tier():
            console_warning(
                "api trace",
                "ROCTX bindings not found; skipping triton instrumentation.",
            )
            return
        _register_framework_root()
        patch_triton_launcher()


register(TritonBackend())
