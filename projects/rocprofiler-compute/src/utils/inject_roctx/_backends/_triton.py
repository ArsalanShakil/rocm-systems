# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for Triton.

Wraps the Triton kernel-launch entry points (``JITFunction.run`` and
``CompiledKernel.run`` / ``CompiledKernel.__call__``) so that Triton and
Inductor kernel launches appear in ROCTX markers.
"""

import importlib.util
import inspect
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from utils.inject_roctx import _core
from utils.inject_roctx._core import (
    MAX_ARG_ITEMS,
    _pop_scope,
    _push_scope,
    args_capture_enabled,
    cap_args,
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


def _format_triton_arg(obj: object) -> str:
    """Render one launch arg: ``dtype[d0xd1]`` for tensors, else its value."""
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None and not isinstance(obj, (str, bytes)):
        try:
            dims = "x".join(str(int(d)) for d in shape)
        except Exception:
            dims = "?"
        return f"{str(dtype).replace('torch.', '')}[{dims}]"
    if isinstance(obj, bool) or isinstance(obj, (int, float)):
        return repr(obj)
    if isinstance(obj, str):
        return repr(obj[:32])
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(_format_triton_arg(o) for o in obj[:8]) + "]"
    return type(obj).__name__


# Launch kwargs that describe runtime geometry rather than kernel arguments.
_TRITON_SKIP_KWARGS = frozenset({"grid", "warmup", "stream", "num_warps", "num_stages"})


def _build_triton_args(
    self_obj: object,
    call_args: tuple[object, ...],
    call_kwargs: dict[str, object],
) -> str:
    """Build the raw (unencoded) leaf-args blob for a Triton kernel launch.

    Positional args are labelled with kernel parameter names when available.
    """
    if not args_capture_enabled():
        return ""
    try:
        names: Optional[list[Any]] = None
        params = getattr(self_obj, "params", None)
        if params:
            try:
                names = [getattr(p, "name", None) for p in params]
            except Exception:
                names = None
        if names is None:
            names = getattr(self_obj, "arg_names", None)

        parts: list[str] = []
        for i, value in enumerate(call_args[:MAX_ARG_ITEMS]):
            label = names[i] if names and i < len(names) and names[i] else None
            rendered = _format_triton_arg(value)
            parts.append(f"{label}={rendered}" if label else rendered)
        for key, value in list(call_kwargs.items())[:MAX_ARG_ITEMS]:
            if key in _TRITON_SKIP_KWARGS:
                continue
            parts.append(f"{key}={_format_triton_arg(value)}")
        return cap_args("(" + ", ".join(parts) + ")")
    except Exception:
        return ""


def _run_with_marker(
    self_obj: object,
    marker_prefix: str,
    thunk: Callable[[], Any],
    call_args: tuple[object, ...] = (),
    call_kwargs: Optional[dict[str, object]] = None,
) -> object:
    """Run ``thunk`` inside a ROCTX range; nested launches reuse the outer range."""
    if _in_launch():
        return thunk()
    kernel_name = _extract_kernel_name(self_obj)
    location = resolve_user_caller_location()
    op_args = _build_triton_args(self_obj, call_args, call_kwargs or {})
    _thread_local.in_launch = True
    pushed = False
    try:
        _push_scope(
            f"{marker_prefix}.{kernel_name}",
            f"#1@{location}",
            backend=_BACKEND_NAME,
            args=op_args,
        )
        pushed = True
        return thunk()
    finally:
        if pushed:
            _pop_scope()
        _thread_local.in_launch = False


def _wrap_method(
    owner: type, method_name: str, marker_prefix: str, original: Callable[..., Any]
) -> bool:
    @wraps(original)
    def launch_with_roctx(self: object, *args: Any, **kwargs: Any) -> object:
        return _run_with_marker(
            self,
            marker_prefix,
            lambda: original(self, *args, **kwargs),
            args,
            kwargs,
        )

    launch_with_roctx._roctx_wrapped = True
    setattr(owner, method_name, launch_with_roctx)
    return True


def _wrap_property(
    owner: type, method_name: str, marker_prefix: str, prop: property
) -> bool:
    orig_get = prop.fget
    if orig_get is None:
        return False

    def roctx_get(self: object) -> object:
        launcher = orig_get(self)
        if launcher is None or getattr(launcher, "_roctx_launcher", False):
            return launcher

        @wraps(launcher)
        def launch(*args: Any, **kwargs: Any) -> object:
            return _run_with_marker(
                self,
                marker_prefix,
                lambda: launcher(*args, **kwargs),
                args,
                kwargs,
            )

        launch._roctx_launcher = True
        return launch

    roctx_get._roctx_wrapped = True
    setattr(owner, method_name, property(roctx_get, prop.fset, prop.fdel))
    return True


def _wrap_launch(
    owner: type,
    method_name: str,
    marker_prefix: str,
) -> bool:
    """Wrap ``owner.method_name`` (a method or property) with a ROCTX range.

    Idempotent. Returns True when the wrapper is installed or already present.
    """
    attr = inspect.getattr_static(owner, method_name, None)
    if attr is None:
        return False
    if isinstance(attr, property):
        if attr.fget is not None and getattr(attr.fget, "_roctx_wrapped", False):
            return True
    elif getattr(attr, "_roctx_wrapped", False):
        return True

    try:
        if isinstance(attr, property):
            installed = _wrap_property(owner, method_name, marker_prefix, attr)
        else:
            installed = _wrap_method(owner, method_name, marker_prefix, attr)
    except Exception as exc:
        console_warning(
            "api trace",
            f"Could not patch {owner.__name__}.{method_name}: {exc}",
        )
        return False

    if installed:
        console_log(
            "api trace",
            f"Wrapped {owner.__name__}.{method_name} with ROCTX markers",
        )
    return installed


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
