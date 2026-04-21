"""Shared pytest fixtures and helpers for RDC CI tests."""

import os
import subprocess
import signal
import time
import shutil
import pytest


def _find_binary(name):
    """Locate an RDC binary, checking ROCM_DIR first, then PATH."""
    rocm = os.environ.get("ROCM_DIR", "/opt/rocm")
    candidate = os.path.join(rocm, "bin", name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    if found:
        return found
    return None


@pytest.fixture(scope="session")
def rocm_dir():
    return os.environ.get("ROCM_DIR", "/opt/rocm")


@pytest.fixture(scope="session")
def rdcd_path(rocm_dir):
    path = _find_binary("rdcd")
    if path is None:
        pytest.skip("rdcd binary not found")
    return path


@pytest.fixture(scope="session")
def rdci_path(rocm_dir):
    path = _find_binary("rdci")
    if path is None:
        pytest.skip("rdci binary not found")
    return path


@pytest.fixture(scope="session")
def rdctst_path(rocm_dir):
    """Locate the rdctst GoogleTest binary."""
    candidates = [
        os.path.join(rocm_dir, "share", "rdc", "rdctst"),
        os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdctst"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    path = _find_binary("rdctst")
    if path is None:
        pytest.skip("rdctst binary not found")
    return path


@pytest.fixture(scope="session")
def has_gpu():
    """Return True if at least one AMD GPU is accessible."""
    try:
        vendor_path = "/sys/class/drm/card0/device/vendor"
        if os.path.isfile(vendor_path):
            with open(vendor_path) as f:
                if "1002" in f.read():
                    return True
    except Exception:
        pass
    return False


def _wait_for_port(port, timeout=10):
    """Block until something is listening on *port* or timeout expires."""
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("localhost", port)) == 0:
                return True
        time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def rdcd_server(rdcd_path, has_gpu):
    """Start rdcd in unauthenticated mode for the test session.

    Yields the subprocess.Popen handle.  Cleans up on teardown.
    Skips if no GPU is present (daemon won't be useful).
    """
    if not has_gpu:
        pytest.skip("No AMD GPU detected -- skipping daemon tests")

    env = os.environ.copy()
    proc = subprocess.Popen(
        [rdcd_path, "-u"],  # unauthenticated mode
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not _wait_for_port(50051, timeout=15):
        proc.kill()
        pytest.fail("rdcd did not start within 15 s")

    yield proc

    # Teardown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_rdci(rdci_path, *args, timeout=30):
    """Run rdci with the given arguments and return CompletedProcess."""
    cmd = [rdci_path, "-u"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="session")
def python_binding_dir():
    """Return path to python_binding/ directory."""
    base = os.environ.get(
        "RDC_SRC_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "python_binding")
    )
    base = os.path.realpath(base)
    if not os.path.isdir(base):
        pytest.skip("python_binding directory not found")
    return base
