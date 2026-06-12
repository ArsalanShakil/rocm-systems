# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the ``utils.inject_roctx`` public surface:
``install_global_wraps``, ``_backends.install_many``, ``TritonBackend``, and
``_core._push_scope`` / ``_pop_scope``."""

import importlib
import sys
import types

import common  # noqa: F401
import pytest

# ---------------------------------------------------------------------------
# install_global_wraps
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_install(monkeypatch):
    """Replace ``_backends.install_many`` with a recorder."""
    from utils.inject_roctx import _backends as backends_pkg

    calls: list[list[str]] = []

    def _record(names):
        calls.append(list(names))

    monkeypatch.setattr(backends_pkg, "install_many", _record)
    return calls


def test_install_global_wraps_empty_inputs_are_noop(captured_install):
    from utils.inject_roctx import install_global_wraps

    install_global_wraps("")
    install_global_wraps([])
    assert captured_install == []


def test_install_global_wraps_single_name(captured_install):
    from utils.inject_roctx import install_global_wraps

    install_global_wraps("torch")
    assert captured_install == [["torch"]]


def test_install_global_wraps_comma_split_with_whitespace(captured_install):
    from utils.inject_roctx import install_global_wraps

    install_global_wraps("torch , , triton")
    assert captured_install == [["torch", "triton"]]


def test_install_global_wraps_iterable_input(captured_install):
    from utils.inject_roctx import install_global_wraps

    install_global_wraps(["torch", "triton"])
    assert captured_install == [["torch", "triton"]]


def test_install_global_wraps_api_alias_expands(captured_install):
    from utils.inject_roctx import install_global_wraps

    install_global_wraps("api")
    assert captured_install == [["torch", "triton"]]


def test_install_global_wraps_api_alongside_explicit_name(captured_install):
    from utils.inject_roctx import install_global_wraps

    install_global_wraps("api,torch")
    assert captured_install == [["torch", "triton", "torch"]]


# ---------------------------------------------------------------------------
# _backends.install_many
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry(monkeypatch):
    """Provide an isolated registry for ``install_many`` tests."""
    from utils.inject_roctx import _backends as backends_pkg

    monkeypatch.setattr(backends_pkg, "_REGISTRY", {})
    return backends_pkg


def _make_backend(name, install_fn=None):
    backend = types.SimpleNamespace()
    backend.name = name
    backend.install = install_fn or (lambda: None)
    return backend


def test_install_many_invokes_registered_backends(fresh_registry):
    calls: list[str] = []
    fresh_registry.register(_make_backend("alpha", lambda: calls.append("alpha")))
    fresh_registry.register(_make_backend("beta", lambda: calls.append("beta")))

    fresh_registry.install_many(["alpha", "beta"])
    assert calls == ["alpha", "beta"]


def test_install_many_dedupes_duplicate_names(fresh_registry):
    calls: list[str] = []
    fresh_registry.register(_make_backend("alpha", lambda: calls.append("alpha")))

    fresh_registry.install_many(["alpha", "alpha", "alpha"])
    assert calls == ["alpha"]


def test_install_many_continues_after_backend_failure(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    other_calls: list[str] = []
    fresh_registry.register(
        _make_backend("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    fresh_registry.register(_make_backend("good", lambda: other_calls.append("good")))

    fresh_registry.install_many(["bad", "good"])
    assert other_calls == ["good"]
    assert any("bad" in str(args[1]) for args in warnings if len(args) > 1)


def test_install_many_warns_on_unknown_backend(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    fresh_registry.install_many(["does_not_exist_zzz"])
    assert any(
        "does_not_exist_zzz" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_install_many_warns_when_module_does_not_register(fresh_registry, monkeypatch):
    warnings: list[tuple] = []
    monkeypatch.setattr("utils.logger.console_warning", lambda *a: warnings.append(a))

    pkg_name = fresh_registry.__name__
    fake_name = f"{pkg_name}._ghost"
    sys.modules[fake_name] = types.ModuleType(fake_name)
    try:
        fresh_registry.install_many(["ghost"])
    finally:
        sys.modules.pop(fake_name, None)

    assert any("did not register" in str(args[1]) for args in warnings if len(args) > 1)


# ---------------------------------------------------------------------------
# TritonBackend
# ---------------------------------------------------------------------------


def test_triton_backend_skips_when_triton_missing(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    real_find_spec = importlib.util.find_spec

    def _no_triton(name, *args, **kwargs):
        if name == "triton":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _no_triton)

    warnings: list[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert any(
        "Triton is not installed" in str(args[1]) for args in warnings if len(args) > 1
    )


def test_triton_backend_wraps_compiled_kernel_call(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    pushes: list[tuple] = []
    pops: list[None] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="", args="": pushes.append((marker, ctx, backend)),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: pops.append(None))

    class FakeKernel:
        name = "my_kernel"

        def __call__(self, *a, **kw):
            return ("ran", a, kw)

    monkeypatch.setattr(triton_backend, "CompiledKernel", FakeKernel)
    triton_backend.patch_triton_launcher()

    out = FakeKernel()(1, x=2)
    assert out == ("ran", (1,), {"x": 2})
    assert len(pushes) == 1
    marker, ctx, backend = pushes[0]
    assert marker == "triton.CompiledKernel.my_kernel"
    assert ctx.startswith("#1@")
    assert backend == "triton"
    assert pops == [None]


def test_triton_backend_kernel_name_fallbacks(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="", args="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)

    class KernelWithDictMeta:
        metadata = {"name": "from_meta"}

        def __call__(self):
            pass

    class KernelNoName:
        def __call__(self):
            pass

    for cls, expected in (
        (KernelWithDictMeta, "triton.CompiledKernel.from_meta"),
        (KernelNoName, "triton.CompiledKernel.<triton_kernel>"),
    ):
        monkeypatch.setattr(triton_backend, "CompiledKernel", cls)
        triton_backend.patch_triton_launcher()
        cls()()
        assert pushes[-1] == expected


def test_triton_backend_patch_is_idempotent(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    monkeypatch.setattr(triton_backend, "_push_scope", lambda *a, **k: None)
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)

    class FakeKernel:
        name = "k"

        def __call__(self):
            pass

    monkeypatch.setattr(triton_backend, "CompiledKernel", FakeKernel)
    triton_backend.patch_triton_launcher()
    first = FakeKernel.__call__
    triton_backend.patch_triton_launcher()
    assert FakeKernel.__call__ is first


def test_triton_backend_wraps_compiled_kernel_run(monkeypatch):
    """CompiledKernel.run() is wrapped in preference to __call__."""
    from utils.inject_roctx._backends import _triton as triton_backend

    pushes: list[tuple] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="", args="": pushes.append((marker, backend)),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend, "JITFunction", None)

    class FakeCompiledKernel:
        name = "rk"

        def run(self, *a, **kw):
            return "ran"

    monkeypatch.setattr(triton_backend, "CompiledKernel", FakeCompiledKernel)
    triton_backend.patch_triton_launcher()

    assert FakeCompiledKernel().run() == "ran"
    assert pushes == [("triton.CompiledKernel.rk", "triton")]


def test_triton_backend_wraps_jitfunction_run(monkeypatch):
    """JITFunction.run is wrapped for eager launches."""
    from utils.inject_roctx._backends import _triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="", args="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    monkeypatch.setattr(triton_backend, "CompiledKernel", None)

    class FakeJIT:
        def __init__(self):
            self.fn = types.SimpleNamespace(__name__="add_kernel")

        def run(self, *a, **kw):
            return "launched"

    monkeypatch.setattr(triton_backend, "JITFunction", FakeJIT)
    triton_backend.patch_triton_launcher()

    assert FakeJIT().run() == "launched"
    assert pushes == ["triton.JITFunction.add_kernel"]


def test_triton_backend_reentrancy_dedups_nested_launch(monkeypatch):
    """Nested JITFunction.run and CompiledKernel.run emit one marker."""
    from utils.inject_roctx._backends import _triton as triton_backend

    pushes: list[str] = []
    monkeypatch.setattr(
        triton_backend,
        "_push_scope",
        lambda marker, ctx, backend="", args="": pushes.append(marker),
    )
    monkeypatch.setattr(triton_backend, "_pop_scope", lambda: None)
    # Reset the per-thread guard.
    if hasattr(triton_backend._thread_local, "in_launch"):
        del triton_backend._thread_local.in_launch

    class FakeCompiledKernel:
        name = "inner"

        def run(self, *a, **kw):
            return "inner_ran"

    class FakeJIT:
        name = "outer"

        def __init__(self, compiled):
            self._compiled = compiled

        def run(self, *a, **kw):
            return self._compiled.run()

    monkeypatch.setattr(triton_backend, "CompiledKernel", FakeCompiledKernel)
    monkeypatch.setattr(triton_backend, "JITFunction", FakeJIT)
    triton_backend.patch_triton_launcher()

    compiled = FakeCompiledKernel()
    out = FakeJIT(compiled).run()

    assert out == "inner_ran"
    assert pushes == ["triton.JITFunction.outer"]


def test_triton_backend_registers_framework_root(monkeypatch):
    """install() registers triton's package directory as a framework root."""
    from utils.inject_roctx._backends import _triton as triton_backend

    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(triton_backend, "patch_triton_launcher", lambda: None)

    fake_triton = types.ModuleType("triton")
    fake_triton.__file__ = "/opt/fake/triton/__init__.py"
    monkeypatch.setitem(sys.modules, "triton", fake_triton)

    roots: list[str] = []
    monkeypatch.setattr(
        triton_backend._core, "add_framework_root", lambda p: roots.append(p)
    )

    triton_backend.TritonBackend().install()
    assert roots == ["/opt/fake/triton"]


def test_triton_backend_skips_when_python_tier_unavailable(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    monkeypatch.setattr(triton_backend, "_resolve_triton", lambda: True)
    monkeypatch.setattr(triton_backend._core, "ensure_python_tier", lambda: False)

    patched: list[bool] = []
    monkeypatch.setattr(
        triton_backend, "patch_triton_launcher", lambda: patched.append(True)
    )
    warnings: list[tuple] = []
    monkeypatch.setattr(
        triton_backend, "console_warning", lambda *a: warnings.append(a)
    )

    triton_backend.TritonBackend().install()
    assert patched == []
    assert any("ROCTX bindings not found" in str(a[1]) for a in warnings if len(a) > 1)


def test_ensure_python_tier_short_circuits_when_already_configured(monkeypatch):
    from utils.inject_roctx import _core

    saved_push, saved_pop = _core._range_push, _core._range_pop
    saved_ready = _core._python_tier_ready
    try:
        _core.set_python_tier_io(lambda _s: None, lambda: None)

        def _boom(*_a, **_k):
            raise AssertionError("roctx import should be skipped")

        monkeypatch.setattr(_core.importlib, "import_module", _boom)
        assert _core.ensure_python_tier() is True
    finally:
        _core._range_push, _core._range_pop = saved_push, saved_pop
        _core._python_tier_ready = saved_ready


def test_extract_kernel_name_prefers_attr_then_meta_then_fn():
    from utils.inject_roctx._backends import _triton as triton_backend

    named = types.SimpleNamespace(name="direct")
    assert triton_backend._extract_kernel_name(named) == "direct"

    meta = types.SimpleNamespace(metadata={"name": "meta_name"})
    assert triton_backend._extract_kernel_name(meta) == "meta_name"

    via_fn = types.SimpleNamespace(fn=types.SimpleNamespace(__name__="fn_name"))
    assert triton_backend._extract_kernel_name(via_fn) == "fn_name"

    assert (
        triton_backend._extract_kernel_name(types.SimpleNamespace())
        == "<triton_kernel>"
    )


# ---------------------------------------------------------------------------
# _core push/pop
# ---------------------------------------------------------------------------


@pytest.fixture
def core_with_python_tier():
    """Return ``_core`` wired to in-memory push/pop sinks with empty stacks."""
    from utils.inject_roctx import _core

    pushed: list[str] = []
    popped: list[None] = []
    _core.set_python_tier_io(push=pushed.append, pop=lambda: popped.append(None))
    _core.set_native_tier_hook(None)
    for attr in ("marker_stack", "context_stack", "tier_stack"):
        if hasattr(_core._thread_local, attr):
            delattr(_core._thread_local, attr)
    return _core, pushed, popped


def test_push_scope_appends_backend_suffix(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", backend="torch")
    assert pushed == ["op:#1@x:1|torch"]


def test_push_scope_omits_suffix_when_backend_empty(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1")
    assert pushed == ["op:#1@x:1"]


def test_push_scope_routes_to_native_tier_when_active(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    seen: list[tuple] = []

    class Hook:
        def active(self):
            return True

        def push(self, marker, context, backend, args=""):
            seen.append((marker, context, backend))
            return True

        def pop(self):
            pass

    core.set_native_tier_hook(Hook())
    try:
        core._push_scope("op", "#1@x:1", backend="torch")
    finally:
        core.set_native_tier_hook(None)

    assert seen == [("op", "#1@x:1", "torch")]
    assert pushed == []


def test_pop_scope_routes_each_frame_to_its_originating_tier(core_with_python_tier):
    core, pushed, popped = core_with_python_tier

    native_pops: list[None] = []

    class Hook:
        active_flag = True

        def active(self):
            return self.active_flag

        def push(self, marker, context, backend, args=""):
            return True

        def pop(self):
            native_pops.append(None)

    hook = Hook()
    core.set_native_tier_hook(hook)
    try:
        core._push_scope("native_op", "#1@x:1", backend="torch")
        hook.active_flag = False
        core._push_scope("py_op", "#2@x:2", backend="torch")
        core._pop_scope()
        core._pop_scope()
    finally:
        core.set_native_tier_hook(None)

    assert pushed == ["native_op/py_op:#1@x:1/#2@x:2|torch"]
    assert popped == [None]
    assert native_pops == [None]


# ---------------------------------------------------------------------------
# Marker name percent-encoding
# ---------------------------------------------------------------------------


def test_push_scope_percent_encodes_slash_and_percent(core_with_python_tier):
    """``_push_scope`` encodes '/' as %2F and '%' as %25 within a marker name."""
    core, pushed, _ = core_with_python_tier

    core._push_scope("a/b%c%2Fd", "#1@x:1")

    assert pushed == ["a%2Fb%25c%252Fd:#1@x:1"]


def test_marker_encoding_round_trips_through_build_call_trees(core_with_python_tier):
    """Names encoded by ``_push_scope`` are restored by ``build_call_trees``."""
    import pandas as pd

    from utils.utils_analysis import build_call_trees

    core, pushed, _ = core_with_python_tier

    outer = "Torch-Compiled Region: 0/0"
    inner = "kernel%name_with_%2F_literal"

    core._push_scope(outer, "#1@a.py:1")
    core._push_scope(inner, "#2@b.py:2")

    # The operator path precedes the ':' that separates it from the context.
    expected_encoded = "Torch-Compiled Region: 0%2F0/kernel%25name_with_%252F_literal"
    assert pushed[-1].startswith(expected_encoded + ":")

    df = pd.DataFrame({
        "Operator_Name": [expected_encoded],
        "Kernel_Name": ["my_kernel"],
    })
    trees = build_call_trees(df)

    (root,) = trees.values()
    assert outer in root.children, list(root.children)
    outer_node = root.children[outer]
    assert inner in outer_node.children, list(outer_node.children)
    assert "my_kernel" in outer_node.children[inner].kernels


# ---------------------------------------------------------------------------
# Operator args capture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "types=Tensor;shapes=[[2, 2]]",
        "a|b%c\nd\re",
        "(float32[2x3], dim=1)",
        "100% | done",
        "a;b;c",
        "%3B literal and ; raw",
    ],
)
def test_encode_args_round_trips(raw):
    from utils.inject_roctx import _core

    encoded = _core._encode_args(raw)
    # The encoded form must not contain the reserved delimiters or newlines.
    assert "|" not in encoded
    assert ";" not in encoded
    assert "\n" not in encoded
    assert "\r" not in encoded
    assert _core._decode_args(encoded) == raw


def test_push_scope_appends_args_segment_before_backend(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", backend="torch", args="(f32[2x2])")
    assert pushed == ["op:#1@x:1|args=(f32[2x2])|torch"]


def test_push_scope_args_segment_without_backend(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", args="(f32[2x2])")
    assert pushed == ["op:#1@x:1|args=(f32[2x2])"]


def test_push_scope_encodes_pipe_in_args(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    core._push_scope("op", "#1@x:1", backend="torch", args="a|b")
    # The '|' inside args is encoded so the trailing backend stays parseable.
    assert pushed == ["op:#1@x:1|args=a%7Cb|torch"]


def test_push_scope_forwards_args_to_native_tier(core_with_python_tier):
    core, pushed, _ = core_with_python_tier

    seen: list[tuple] = []

    class Hook:
        def active(self):
            return True

        def push(self, marker, context, backend, args=""):
            seen.append((marker, context, backend, args))
            return True

        def pop(self):
            pass

    core.set_native_tier_hook(Hook())
    try:
        core._push_scope("op", "#1@x:1", backend="torch", args="(f32[2x2])")
    finally:
        core.set_native_tier_hook(None)

    assert seen == [("op", "#1@x:1", "torch", "(f32[2x2])")]
    assert pushed == []


def test_args_capture_env_gate(monkeypatch):
    from utils.inject_roctx import _core

    monkeypatch.delenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARGS", raising=False)
    assert _core.args_capture_enabled() is True
    monkeypatch.setenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARGS", "0")
    assert _core.args_capture_enabled() is False
    monkeypatch.setenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARGS", "off")
    assert _core.args_capture_enabled() is False

    monkeypatch.delenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARG_VALUES", raising=False)
    assert _core.args_values_enabled() is False
    monkeypatch.setenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARG_VALUES", "1")
    assert _core.args_values_enabled() is True


def test_cap_args_truncates_long_blobs():
    from utils.inject_roctx import _core

    long_blob = "x" * (_core.MAX_ARGS_LEN + 50)
    capped = _core.cap_args(long_blob)
    assert capped.endswith("...")
    assert len(capped) == _core.MAX_ARGS_LEN + len("...")
    assert _core.cap_args("short") == "short"


def test_triton_build_args_tensor_and_scalar(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    monkeypatch.delenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARGS", raising=False)

    fake_tensor = types.SimpleNamespace(shape=(2, 3), dtype="torch.float32")
    params = [
        types.SimpleNamespace(name="x_ptr"),
        types.SimpleNamespace(name="n_elements"),
    ]
    self_obj = types.SimpleNamespace(params=params)

    blob = triton_backend._build_triton_args(
        self_obj, (fake_tensor, 1024), {"grid": (8,), "BLOCK_SIZE": 256}
    )
    assert "x_ptr=float32[2x3]" in blob
    assert "n_elements=1024" in blob
    # grid is a runtime geometry kwarg and must be skipped.
    assert "grid" not in blob
    assert "BLOCK_SIZE=256" in blob


def test_triton_build_args_respects_gate(monkeypatch):
    from utils.inject_roctx._backends import _triton as triton_backend

    monkeypatch.setenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARGS", "0")
    blob = triton_backend._build_triton_args(
        types.SimpleNamespace(params=None), (1, 2), {}
    )
    assert blob == ""


def test_torch_build_dispatch_args_formats(monkeypatch):
    from utils.inject_roctx._backends import _torch as torch_backend

    monkeypatch.delenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARGS", raising=False)
    monkeypatch.delenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARG_VALUES", raising=False)

    # Stand in for torch.Tensor so the formatter takes the tensor branch.
    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    fake_torch = types.SimpleNamespace(Tensor=FakeTensor)
    monkeypatch.setattr(torch_backend, "torch", fake_torch)

    t = FakeTensor((4, 8), "torch.float16")
    blob = torch_backend.build_dispatch_args((t,), {"dim": 1})
    assert "float16[4x8]" in blob
    # dim value is hidden unless value capture is enabled.
    assert "dim=int" in blob

    monkeypatch.setenv("ROCPROFCOMPUTE_ROCTX_CAPTURE_ARG_VALUES", "1")
    blob_values = torch_backend.build_dispatch_args((t,), {"dim": 1})
    assert "dim=1" in blob_values
