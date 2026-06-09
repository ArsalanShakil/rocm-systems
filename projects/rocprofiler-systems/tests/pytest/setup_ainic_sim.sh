#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# setup_ainic_sim.sh — build the fake sysfs AI NIC tree and start the
# background hw_counter simulator.  Keep running until Ctrl-C / SIGTERM.
#
# Run this in a SEPARATE terminal BEFORE running the AI NIC pytest tests:
#
#   Terminal 1 (setup):
#       cd <repo>/projects/rocprofiler-systems/tests/pytest
#       ./setup_ainic_sim.sh
#       # Copy the "export SMI_NIC_SYSFS_ROOT=..." line that is printed.
#
#   Terminal 2 (tests):
#       export SMI_NIC_SYSFS_ROOT=<path printed above>
#       cd <repo>/projects/rocprofiler-systems/tests/pytest
#       pytest -m ainic test_ainic_perf.py -v
#
# The script creates the fake sysfs tree under /tmp/ainic-sim-<PID> and
# removes it automatically when it exits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="/tmp/ainic-sim-$$"

cleanup() {
    echo ""
    echo "[setup_ainic_sim] Stopping simulator and cleaning up …"
    if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
        kill "$SIM_PID"
        wait "$SIM_PID" 2>/dev/null || true
    fi
    rm -rf "$SIM_ROOT"
    echo "[setup_ainic_sim] Done."
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------
# Build the fake sysfs tree and retrieve the hw_counters path
# -----------------------------------------------------------------------
echo "[setup_ainic_sim] Creating fake sysfs under $SIM_ROOT …"

HW_COUNTERS_DIR=$(python3 - <<PYEOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from pathlib import Path
import fake_sysfs
hw = fake_sysfs.create(Path('$SIM_ROOT'))
print(hw)
PYEOF
)

echo "[setup_ainic_sim] Fake sysfs created."
echo "[setup_ainic_sim] hw_counters dir: $HW_COUNTERS_DIR"

# -----------------------------------------------------------------------
# Start the NIC simulator in the background
# -----------------------------------------------------------------------
echo "[setup_ainic_sim] Starting NIC counter simulator …"
python3 "$SCRIPT_DIR/nic_simulator.py" "$HW_COUNTERS_DIR" &
SIM_PID=$!
echo "[setup_ainic_sim] Simulator PID: $SIM_PID"

# -----------------------------------------------------------------------
# Print the environment variable for the test terminal
# -----------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  AI NIC simulation is ACTIVE."
echo ""
echo "  In your test terminal, run:"
echo ""
echo "    export SMI_NIC_SYSFS_ROOT=$SIM_ROOT"
echo ""
echo "  Then run the AI NIC tests:"
echo ""
echo "    pytest -m ainic test_ainic_perf.py -v"
echo "============================================================"
echo ""
echo "[setup_ainic_sim] Press Ctrl-C to stop the simulation."

# -----------------------------------------------------------------------
# Keep alive until the user interrupts
# -----------------------------------------------------------------------
while kill -0 "$SIM_PID" 2>/dev/null; do
    sleep 2
done

echo "[setup_ainic_sim] Simulator exited unexpectedly (PID $SIM_PID)."
exit 1
