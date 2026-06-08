# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_ainic_sim.py — end-to-end simulation test for AI NIC (Pensando Pollara)
metrics collection in rocprofiler-systems.

The test creates a fake sysfs tree, starts a background simulator that
increments the hw_counter files, then runs rocprof-sys-sample with
SMI_NIC_SYSFS_ROOT pointing at the fake tree.  It asserts that:

  1. rocprof-sys-sample exits successfully.
  2. At least one NIC device appears in the output.
  3. The collected RDMA counters are non-zero (proving the simulator's
     updates were picked up by the sampler during the run).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate helper scripts relative to this test file
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_FAKE_SYSFS_PY   = _HERE / "fake_sysfs.py"
_NIC_SIM_PY      = _HERE / "nic_simulator.py"

pytestmark = [pytest.mark.ainic_sim]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_sysfs_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the fake sysfs tree once per test session."""
    # Import fake_sysfs from the same directory as this test file
    sys.path.insert(0, str(_HERE))
    import fake_sysfs  # noqa: PLC0415

    root = tmp_path_factory.mktemp("fake-sysfs")
    fake_sysfs.create(root)
    return root


@pytest.fixture(scope="session")
def hw_counters_dir(fake_sysfs_root: Path) -> Path:
    """Return the hw_counters directory inside the fake sysfs tree."""
    import fake_sysfs  # noqa: PLC0415

    return (
        fake_sysfs_root
        / "sys/devices/pci0000:e0"
        / fake_sysfs.BRIDGE_BDF
        / fake_sysfs.PORT_BDF
        / "infiniband"
        / fake_sysfs.IB_DEV
        / "ports"
        / fake_sysfs.IB_PORT
        / "hw_counters"
    )


@pytest.fixture(scope="session")
def nic_simulator(hw_counters_dir: Path):
    """Start the NIC simulator subprocess; stop it after the session."""
    proc = subprocess.Popen(
        [sys.executable, str(_NIC_SIM_PY), str(hw_counters_dir), "--interval", "0.05"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Give the simulator a moment to start and write the first tick
    time.sleep(0.2)
    assert proc.poll() is None, "nic_simulator failed to start"
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Helper: build the environment for rocprof-sys-sample
# ---------------------------------------------------------------------------

def _make_env(fake_sysfs_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["SMI_NIC_SYSFS_ROOT"] = str(fake_sysfs_root)
    # Enable AI NIC PMC sampling
    env.setdefault("ROCPROFSYS_USE_PROCESS_SAMPLING", "ON")
    env.setdefault("ROCPROFSYS_SAMPLING_FREQ", "10")       # 10 Hz
    env.setdefault("ROCPROFSYS_USE_PID", "OFF")
    env.setdefault("ROCPROFSYS_TRACE", "OFF")
    return env


def _find_ainic_info_binary() -> "str | None":
    """Return path to amd_smi_ainic_info, or None if not found."""
    found = shutil.which("amd_smi_ainic_info")
    if found:
        return found
    here = Path(__file__).resolve()
    repo_root = here.parents[4]   # tests/pytest -> rocprofiler-systems -> projects -> repo
    for candidate in [
        repo_root / "build" / "projects" / "amdsmi" / "example" / "amd_smi_ainic_info",
        repo_root / "build-release" / "projects" / "amdsmi" / "example" / "amd_smi_ainic_info",
        repo_root / "projects" / "amdsmi" / "example" / "build" / "amd_smi_ainic_info",
    ]:
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAINICSim:
    """Tests that verify AI NIC simulation works end-to-end."""

    def test_fake_sysfs_structure(self, fake_sysfs_root: Path, hw_counters_dir: Path) -> None:
        """Verify the fake sysfs tree has the expected structure."""
        import fake_sysfs  # noqa: PLC0415

        # PCI devices symlinks exist
        assert (fake_sysfs_root / "sys/bus/pci/devices" / fake_sysfs.BRIDGE_BDF).is_symlink()
        assert (fake_sysfs_root / "sys/bus/pci/devices" / fake_sysfs.PORT_BDF).is_symlink()

        # Bridge has correct PCI IDs
        bridge_dev = (
            fake_sysfs_root / "sys/devices/pci0000:e0" / fake_sysfs.BRIDGE_BDF
        )
        assert bridge_dev.is_dir()
        assert bridge_dev / "vendor"
        assert (bridge_dev / "vendor").read_text().strip() == fake_sysfs.VENDOR_ID_HEX
        assert (bridge_dev / "device").read_text().strip() == fake_sysfs.BRIDGE_DEV_HEX

        # Port has correct PCI IDs
        port_dev = bridge_dev / fake_sysfs.PORT_BDF
        assert (port_dev / "vendor").read_text().strip() == fake_sysfs.VENDOR_ID_HEX
        assert (port_dev / "device").read_text().strip() == fake_sysfs.PORT_DEV_HEX

        # hw_counters directory exists and all counters are present
        assert hw_counters_dir.is_dir()
        for counter in fake_sysfs.HW_COUNTERS:
            assert (hw_counters_dir / counter).exists(), f"Missing counter: {counter}"

        # Net interface directory and device symlink exist
        net_iface = fake_sysfs_root / "sys/class/net" / fake_sysfs.IFACE
        assert net_iface.is_dir()
        assert (net_iface / "device").is_symlink()

        # Downstream symlink resolves such that canonical path contains bridge BDF
        import fake_sysfs as _fs
        port_symlink = fake_sysfs_root / "sys/bus/pci/devices" / _fs.PORT_BDF
        canonical = port_symlink.resolve()
        assert f"/{_fs.BRIDGE_BDF}/" in str(canonical), (
            f"downstream_port() check would fail: '{_fs.BRIDGE_BDF}' not in '{canonical}'"
        )

    def test_nic_simulator_increments_counters(
        self,
        nic_simulator: subprocess.Popen,
        hw_counters_dir: Path,
    ) -> None:
        """Verify the simulator is actually incrementing hw_counter values."""
        import fake_sysfs  # noqa: PLC0415

        counter_path = hw_counters_dir / "rx_rdma_ucast_bytes"
        before = int(counter_path.read_text().strip())
        time.sleep(0.3)
        after = int(counter_path.read_text().strip())
        assert after > before, (
            f"rx_rdma_ucast_bytes did not increase: before={before}, after={after}"
        )

    @staticmethod
    def _rocprof_sys_sample_works() -> bool:
        """Return True only if rocprof-sys-sample is present and doesn't crash."""
        binary = shutil.which("rocprof-sys-sample")
        if binary is None:
            return False
        try:
            # Quick sanity run: just ask for help/version — exits fast, no GPU needed
            r = subprocess.run(
                [binary, "--help"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    @pytest.mark.skipif(
        not __import__('os').environ.get("AINIC_TEST_ROCPROF", ""),
        reason=(
            "rocprof-sys-sample test skipped: set AINIC_TEST_ROCPROF=1 and "
            "ensure rocprof-sys-sample is built against the patched amdsmi."
        ),
    )
    def test_rocprof_sys_sample_collects_nic_metrics(
        self,
        fake_sysfs_root: Path,
        nic_simulator: subprocess.Popen,
        tmp_path: Path,
    ) -> None:
        """
        Run rocprof-sys-sample for 2 seconds with the fake sysfs and verify
        that NIC RDMA metrics appear in the output and are non-zero.
        """
        output_dir = tmp_path / "rocprof-out"
        output_dir.mkdir()

        env = _make_env(fake_sysfs_root)

        result = subprocess.run(
            [
                "rocprof-sys-sample",
                "--duration", "2",
                "--output", str(output_dir),
                "--", "sleep", "2",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"rocprof-sys-sample exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        # Look for output files (JSON / CSV) and check for NIC metrics
        output_files = list(output_dir.rglob("*.json")) + list(output_dir.rglob("*.csv"))
        assert output_files, f"No output files found in {output_dir}"

        nic_metric_found = False
        nonzero_value_found = False

        for output_file in output_files:
            content = output_file.read_text()
            # Check for any RDMA metric name in the output
            if any(m in content for m in ["rx_rdma_ucast_bytes", "tx_rdma_ucast_bytes",
                                           "rx_rdma_ucast_pkts", "rx_rdma_cnp_pkts"]):
                nic_metric_found = True
                # Try to find non-zero values
                if output_file.suffix == ".json":
                    try:
                        data = json.loads(content)
                        nonzero_value_found = _has_nonzero_nic_metric(data)
                    except json.JSONDecodeError:
                        pass
                else:
                    # CSV: look for lines with non-zero numbers after metric names
                    for line in content.splitlines():
                        if "rdma" in line.lower():
                            parts = line.split(",")
                            for part in parts:
                                try:
                                    if int(part.strip()) > 0:
                                        nonzero_value_found = True
                                        break
                                except ValueError:
                                    pass

        assert nic_metric_found, (
            "No NIC RDMA metric names found in rocprof-sys-sample output. "
            "The AI NIC simulation may not have been detected."
        )
        assert nonzero_value_found, (
            "NIC RDMA metrics were found but all values are zero. "
            "The simulator updates may not have been picked up."
        )



    # ------------------------------------------------------------------
    # Direct amdsmi simulation test (does not require rocprof-sys-sample)
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        _find_ainic_info_binary() is None,
        reason="amd_smi_ainic_info not found (build amdsmi first)",
    )
    def test_amdsmi_reads_simulated_nic_counters(
        self,
        fake_sysfs_root: Path,
        nic_simulator: subprocess.Popen,
    ) -> None:
        """
        Run amd_smi_ainic_info with SMI_NIC_SYSFS_ROOT pointing at the fake
        sysfs tree while the NIC simulator is running, then verify that:
          - at least one AI NIC device is reported,
          - at least one RDMA hw_counter is listed,
          - at least one counter value is non-zero.
        This is the primary amdsmi-level test for the simulation PR.
        """
        binary = _find_ainic_info_binary()
        assert binary is not None

        env = os.environ.copy()
        env["SMI_NIC_SYSFS_ROOT"] = str(fake_sysfs_root)

        result = subprocess.run(
            [binary],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0, (
            f"amd_smi_ainic_info failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        output = result.stdout
        assert "AI NIC device" in output, (
            "amd_smi_ainic_info produced no AI NIC devices.\n"
            f"stdout:\n{output}"
        )

        rdma_counter_names = [
            "rx_rdma_ucast_bytes", "tx_rdma_ucast_bytes",
            "rx_rdma_ucast_pkts", "tx_rdma_ucast_pkts",
        ]
        assert any(n in output for n in rdma_counter_names), (
            "No RDMA hw_counter names found in amd_smi_ainic_info output.\n"
            f"stdout:\n{output}"
        )

        nonzero = False
        for line in output.splitlines():
            if any(n in line for n in rdma_counter_names):
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        if int(parts[-1].strip()) > 0:
                            nonzero = True
                            break
                    except ValueError:
                        pass

        assert nonzero, (
            "All RDMA counter values are zero — the simulator may not have "
            "been running or the values were not picked up.\n"
            f"stdout:\n{output}"
        )


def _has_nonzero_nic_metric(data: object) -> bool:
    """Recursively search a JSON structure for non-zero NIC RDMA metric values."""
    rdma_keys = {"rx_rdma_ucast_bytes", "tx_rdma_ucast_bytes",
                 "rx_rdma_ucast_pkts", "tx_rdma_ucast_pkts"}
    if isinstance(data, dict):
        for k, v in data.items():
            if k in rdma_keys and isinstance(v, (int, float)) and v > 0:
                return True
            if _has_nonzero_nic_metric(v):
                return True
    elif isinstance(data, list):
        for item in data:
            if _has_nonzero_nic_metric(item):
                return True
    return False
