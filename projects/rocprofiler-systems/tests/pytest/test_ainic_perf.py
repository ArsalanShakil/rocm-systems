# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AI NIC tests using AMD SMI Phase 2 RDMA metrics (amdsmi_get_nic_rdma_dev_info).

These tests verify that rocprofiler-systems correctly collects all 10 AI NIC
RDMA counters per device and writes them to both the Perfetto (.proto) trace
and the ROCpd (.db) database.

---------------------------------------------------------------------------
Running on real hardware
---------------------------------------------------------------------------
When a Pensando Pollara NIC is present (PCI vendor 0x1dd8) the test runs
against the live hardware automatically.  No extra setup is required.

---------------------------------------------------------------------------
Running without hardware — simulation mode
---------------------------------------------------------------------------
The amdsmi library normally reads NIC metrics from Linux sysfs paths such as

    /sys/devices/pci.../infiniband/<dev>/ports/<n>/hw_counters/

The patched amdsmi build used by this project honours an environment variable

    SMI_NIC_SYSFS_ROOT

that redirects every sysfs access to a user-supplied directory tree.  This
lets the tests run on any Linux system that lacks Pensando hardware, including
CI containers.

IMPORTANT — this patch is intentionally NOT part of the upstream amd-smi
repository: the amd-smi team does not accept simulation-only code.  To use
simulation mode you must:

  1. Build amdsmi from the branch that carries the SMI_NIC_SYSFS_ROOT patch
     (projects/amdsmi in this repository).
  2. Install or stage that library so rocprofiler-systems links against it
     rather than the system-wide amd-smi.
  3. Build rocprofiler-systems normally; it will pick up the patched library.
  4. In a separate terminal, run the keep-alive simulation script:

         cd projects/rocprofiler-systems/tests/pytest
         ./setup_ainic_sim.sh

     The script creates a minimal fake sysfs tree, starts a background process
     that continuously increments the hw_counter files (simulating live RDMA
     traffic), and prints the line you need to paste into your test terminal:

         export SMI_NIC_SYSFS_ROOT=/tmp/ainic-sim-<PID>

  5. In the test terminal, paste that export, then run:

         pytest -m ainic test_ainic_perf.py -v

  Leave the simulation terminal running for the full duration of the test run.
  The fake sysfs tree is cleaned up automatically when you press Ctrl-C there.

See AINIC_SIMULATION.md at the repository root for the full build recipe.
"""

from __future__ import annotations

import glob
import os
import shutil
import sqlite3
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.ainic, pytest.mark.network]

# =============================================================================
# Constants
# =============================================================================

# The 10 AI NIC RDMA track names written to the ROCpd (.db) database.
AINIC_ROCPD_TRACK_NAMES = [
    "ainic_rx_rdma_ucast_bytes",
    "ainic_tx_rdma_ucast_bytes",
    "ainic_rx_rdma_ucast_pkts",
    "ainic_tx_rdma_ucast_pkts",
    "ainic_rx_rdma_cnp_pkts",
    "ainic_tx_rdma_cnp_pkts",
    "ainic_tx_rdma_ack_timeout",
    "ainic_resp_tx_pkt_seq_err",
    "ainic_req_rx_pkt_seq_err",
    "ainic_req_rx_impl_nak_seq_err",
]

# Substrings used to match the 10 Perfetto counter track names via LIKE.
# Full name format: "NIC [<device_id>] <METRIC> (S)"
AINIC_PERFETTO_COUNTER_NAMES = [
    "RX RDMA Bytes",
    "TX RDMA Bytes",
    "RX RDMA Packets",
    "TX RDMA Packets",
    "RX CNP Packets",
    "TX CNP Packets",
    "TX ACK TIMEOUT",
    "RESP TX PKT SEQ ERR",
    "REQ RX PKT SEQ ERR",
    "REQ RX IMPL NAK SEQ ERR",
]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ainic_perf_env(rocprof_config) -> dict[str, str]:
    """Environment variables for AI NIC performance tests.

    On systems without Pensando Pollara hardware the tests can be driven by
    the fake-sysfs simulation (see module docstring).  The patched amdsmi
    library recognises ``SMI_NIC_SYSFS_ROOT`` and redirects all sysfs reads
    to the directory it points at.

    The rocprofiler-systems test framework's ``get_fundamental_environment()``
    only auto-forwards variables whose names start with ``ROCPROFSYS_`` or
    ``OMPI_``, so ``SMI_NIC_SYSFS_ROOT`` would be silently dropped before
    reaching the ``rocprof-sys-sample`` child process.  We therefore check for
    it explicitly here and inject it into the environment dictionary when it is
    present.

    Prerequisites for simulation mode
    ----------------------------------
    * amdsmi must be built from the branch that carries the
      ``SMI_NIC_SYSFS_ROOT`` patch (see projects/amdsmi in this repo).
    * ``rocprof-sys-sample`` must be linked against that patched library.
    * ``setup_ainic_sim.sh`` must be running in a separate terminal, and
      ``SMI_NIC_SYSFS_ROOT`` must be exported in the test terminal.

    Note on ``ROCPROFSYS_USE_PROCESS_SAMPLING``
    -------------------------------------------
    ``ROCPROFSYS_USE_AINIC`` is registered in the ``process_sampling``
    category.  When process sampling is ``OFF`` rocprofiler-systems silently
    disables every setting in that category, including AINIC, even if
    ``ROCPROFSYS_USE_AINIC=ON``.  Process sampling must therefore remain
    ``ON``.  CPU PMC noise is suppressed separately via
    ``ROCPROFSYS_SAMPLING_CPUS=none``.
    """
    env = {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_AMD_SMI": "ON",
        "ROCPROFSYS_USE_AINIC": "ON",
        "ROCPROFSYS_SAMPLING_AINICS": "all",
        "ROCPROFSYS_USE_ROCPD": "ON",
        "ROCPROFSYS_SAMPLING_DELAY": "0.05",
    }
    # Propagate SMI_NIC_SYSFS_ROOT when set so that rocprof-sys-sample uses
    # the fake sysfs tree instead of the real /sys hierarchy.  This variable
    # is only recognised by the patched amdsmi build; on an unpatched install
    # it is simply ignored, and the test falls back to real hardware detection.
    sysfs_root = os.environ.get("SMI_NIC_SYSFS_ROOT", "")
    if sysfs_root:
        env["SMI_NIC_SYSFS_ROOT"] = sysfs_root
    return env


def _ainic_hardware_present() -> bool:
    """Return True if a Pensando Pollara NIC is visible in the real sysfs.

    Checks for PCI vendor ID 0x1dd8 (Pensando Systems / AMD) by scanning
    /sys/bus/pci/devices/*/vendor.  This works on the host and inside
    containers that have the host PCI devices bind-mounted.
    """
    for vendor_path in glob.glob("/sys/bus/pci/devices/*/vendor"):
        try:
            with open(vendor_path) as f:
                if f.read().strip() == "0x1dd8":
                    return True
        except OSError:
            pass
    return False


@pytest.fixture(scope="session")
def ainic_available() -> None:
    """Skip the whole session when neither real hardware nor simulation is ready.

    Two conditions allow the test to proceed:

    1. **Real hardware** — a Pensando Pollara NIC is present on the PCI bus.
    2. **Simulation mode** — ``SMI_NIC_SYSFS_ROOT`` is set in the environment,
       meaning the caller has already run ``setup_ainic_sim.sh`` and exported
       the path it printed.  The patched amdsmi will read hw_counters from the
       fake sysfs tree at that path instead of the real /sys hierarchy.

    If neither condition is met the test is skipped with a message that
    explains exactly how to enable simulation mode.
    """
    if os.environ.get("SMI_NIC_SYSFS_ROOT"):
        return  # simulation mode: fake sysfs tree is ready
    if _ainic_hardware_present():
        return  # real Pensando Pollara hardware present
    pytest.skip(
        "No AI NIC hardware detected and SMI_NIC_SYSFS_ROOT is not set.\n"
        "To run in simulation mode:\n"
        "  1. Build amdsmi from the branch with SMI_NIC_SYSFS_ROOT support\n"
        "     (projects/amdsmi in this repo) and install it.\n"
        "  2. Build rocprofiler-systems against that amdsmi.\n"
        "  3. In a separate terminal:\n"
        "       cd projects/rocprofiler-systems/tests/pytest\n"
        "       ./setup_ainic_sim.sh\n"
        "  4. Export the SMI_NIC_SYSFS_ROOT line it prints, then re-run.\n"
        "See AINIC_SIMULATION.md for the full build recipe."
    )


@pytest.fixture
def ainic_download_url_1() -> str:
    """Download URL for the first file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.1/rocprofiler-systems-1.0.1-ubuntu-22.04-ROCm-60400-PAPI-OMPT-Python3.sh"


@pytest.fixture
def ainic_download_url_2() -> str:
    """Download URL for the second file to download."""
    return "https://github.com/ROCm/rocprofiler-systems/releases/download/rocm-6.4.3/rocprofiler-systems-1.0.2-rhel-9.4-PAPI-OMPT-Python3.sh"


# =============================================================================
# Private helpers
# =============================================================================


def _get_ainic_tracks_from_rocpd(db_path: Path) -> set[str]:
    """Return the set of AI NIC track names found in the ROCpd SQLite database.

    Track names are stored in ``rocpd_string_{upid}`` tables. We query every
    such table for strings that start with ``ainic_``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name LIKE 'rocpd_string_%'"
        )
        string_tables = [row[0] for row in cursor.fetchall()]

        found: set[str] = set()
        for table in string_tables:
            try:
                cursor.execute(
                    f"SELECT DISTINCT string FROM {table} WHERE string LIKE 'ainic_%'"
                )
                found.update(row[0] for row in cursor.fetchall())
            except sqlite3.Error:
                pass
        return found
    finally:
        conn.close()


# =============================================================================
# Tests
# =============================================================================


class TestAINIC(RocprofsysTest):
    """Tests for AI NIC performance using AMD SMI Phase 2 RDMA metrics."""

    PERFETTO_PASS_REGEX = [r"perfetto-trace\.proto validated"]
    PERFETTO_FAIL_REGEX = [r"Failure validating.*perfetto-trace\.proto"]

    def test_performance(
        self,
        ainic_available,
        rocprof_config,
        ainic_perf_env,
        ainic_download_url_1,
        ainic_download_url_2,
        test_output_dir,
        subtests,
        record_subtest_failure,
    ):
        target = shutil.which("wget")
        if not target:
            pytest.skip("wget not found")

        download_cmd = [
            "--no-check-certificate",
            ainic_download_url_1,
            ainic_download_url_2,
            "-O",
            str(test_output_dir / "rocprofiler-systems.test.bin"),
        ]
        result = self.run_test(
            "sampling",
            target,
            run_args=download_cmd,
            env=ainic_perf_env,
        )

        self.assert_regex(result)

        # Validate Perfetto .proto: all 10 AI NIC counter track substrings must match
        self.assert_perfetto(
            result,
            counter_names=AINIC_PERFETTO_COUNTER_NAMES,
            pass_regex=self.PERFETTO_PASS_REGEX,
            fail_regex=self.PERFETTO_FAIL_REGEX,
        )

        # Validate ROCpd .db: all 10 AI NIC track names must be present
        subtest_name = "ROCpd AI NIC track validation"
        with subtests.test(subtest_name):
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                record_subtest_failure(subtest_name)
                pytest.fail("ROCpd database (.db) was not created")

            found_tracks = _get_ainic_tracks_from_rocpd(rocpd_file)
            missing = [t for t in AINIC_ROCPD_TRACK_NAMES if t not in found_tracks]
            if missing:
                record_subtest_failure(subtest_name)
                pytest.fail(
                    f"Missing AI NIC tracks in ROCpd database:\n"
                    f"  Missing  : {missing}\n"
                    f"  Found    : {sorted(found_tracks)}\n"
                    f"  Database : {rocpd_file}"
                )
