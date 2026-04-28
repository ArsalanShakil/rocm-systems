#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build and run the rocrtst test suite on the rocjitsu-simulated MI350X (gfx950).
#
# This script:
#   1. Builds rocjitsu (including the KMD interposer library).
#   2. Builds the rocrtst test suite with kernels compiled for gfx950.
#   3. Runs rocrtst under the rocjitsu LD_PRELOAD interposer.
#   4. Generates a test report (text + JUnit XML).
#
# Usage:
#   ./scripts/run-rocrtst.sh [options]
#
# Options:
#   --rocjitsu-build-dir DIR   rocjitsu build directory (default: <project>/build)
#   --rocrtst-build-dir  DIR   rocrtst build directory  (default: <project>/build/rocrtst)
#   --report-dir         DIR   report output directory   (default: <project>/build/rocrtst-report)
#   --skip-rocjitsu-build      skip rebuilding rocjitsu
#   --skip-rocrtst-build       skip rebuilding rocrtst
#   --gtest-filter       FILTER gtest filter pattern     (default: run all tests)
#   --jobs               N     parallel build jobs       (default: nproc)
#   --timeout            SEC   test run timeout          (default: 300)
#   --verbose                  verbose test output
#   -h, --help                 show this help

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROCRTST_SRC="${PROJECT_DIR}/../../projects/rocr-runtime/rocrtst/suites/test_common"
ROCM_ROOT="${ROCM_DIR:-/opt/rocm}"

# Auto-detect the .venv SDK if present (has amd_smi, hsa-runtime64, etc.)
VENV_SDK="${PROJECT_DIR}/.venv/lib/python3.12/site-packages/_rocm_sdk_devel"
if [[ ! -d "$VENV_SDK" ]]; then
    # Try to find it dynamically
    VENV_SDK="$(find "${PROJECT_DIR}/.venv" -maxdepth 5 -name '_rocm_sdk_devel' -type d 2>/dev/null | head -1)"
fi

# Defaults
RJ_BUILD_DIR="${PROJECT_DIR}/build"
ROCRTST_BUILD_DIR="${PROJECT_DIR}/build/rocrtst"
REPORT_DIR="${PROJECT_DIR}/build/rocrtst-report"
GTEST_FILTER=""
SKIP_RJ_BUILD=0
SKIP_ROCRTST_BUILD=0
JOBS="$(nproc)"
VERBOSE=0
TEST_TIMEOUT=300

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocjitsu-build-dir) RJ_BUILD_DIR="$2"; shift 2 ;;
        --rocrtst-build-dir)  ROCRTST_BUILD_DIR="$2"; shift 2 ;;
        --report-dir)         REPORT_DIR="$2"; shift 2 ;;
        --skip-rocjitsu-build) SKIP_RJ_BUILD=1; shift ;;
        --skip-rocrtst-build)  SKIP_ROCRTST_BUILD=1; shift ;;
        --gtest-filter)       GTEST_FILTER="$2"; shift 2 ;;
        --jobs)               JOBS="$2"; shift 2 ;;
        --verbose)            VERBOSE=1; shift ;;
        --timeout)            TEST_TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/s/^# \?//p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Resolve to absolute paths
RJ_BUILD_DIR="$(cd "$RJ_BUILD_DIR" 2>/dev/null && pwd || { mkdir -p "$RJ_BUILD_DIR" && cd "$RJ_BUILD_DIR" && pwd; })"
mkdir -p "$ROCRTST_BUILD_DIR" "$REPORT_DIR"
ROCRTST_BUILD_DIR="$(cd "$ROCRTST_BUILD_DIR" && pwd)"
REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"

# Derived paths
KMD_LIB="${RJ_BUILD_DIR}/lib/rocjitsu/src/rocjitsu/kmd/librocjitsu_kmd.so"
RJ_CONFIG="${PROJECT_DIR}/configs/amdgpu_cdna4_kmd.json"
RJ_SCHEMA="${PROJECT_DIR}/schemas/simulation_config.fbs"
ROCRTST_BIN="${ROCRTST_BUILD_DIR}/rocrtst64"

REPORT_TXT="${REPORT_DIR}/rocrtst-report.txt"
REPORT_XML="${REPORT_DIR}/rocrtst-report.xml"

TARGET_DEVICE="gfx950"

# Tests excluded from the default filter:
#   - Agent_Preload_Latency: calls hsa_amd_agent_preload which is unresolved (crashes).
#   - IPC: requires fork + shared memory between processes (not supported by interposer).
EXCLUDED_TESTS="\
rocrtstPerf.Agent_Preload_Latency:\
rocrtstFunc.IPC"

DEFAULT_FILTER="-${EXCLUDED_TESTS}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# ---------------------------------------------------------------------------
# Step 1: Build rocjitsu (KMD interposer)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[$(timestamp)] rocrtst on rocjitsu MI350X (${TARGET_DEVICE})"
echo "============================================================"

if [[ "$SKIP_RJ_BUILD" -eq 0 ]]; then
    echo ""
    echo "[$(timestamp)] Step 1/3: Building rocjitsu ..."
    pushd "$PROJECT_DIR" > /dev/null
    cmake -B "$RJ_BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DRJ_BUILD_GUI=OFF
    cmake --build "$RJ_BUILD_DIR" -j"$JOBS"
    popd > /dev/null

    if [[ ! -f "$KMD_LIB" ]]; then
        echo "Error: librocjitsu_kmd.so not found at $KMD_LIB" >&2
        exit 1
    fi
    echo "[$(timestamp)] rocjitsu build complete."
else
    echo ""
    echo "[$(timestamp)] Step 1/3: Skipping rocjitsu build (--skip-rocjitsu-build)"
    if [[ ! -f "$KMD_LIB" ]]; then
        echo "Error: librocjitsu_kmd.so not found at $KMD_LIB" >&2
        echo "  Build rocjitsu first or remove --skip-rocjitsu-build." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Build rocrtst for gfx950
# ---------------------------------------------------------------------------
if [[ "$SKIP_ROCRTST_BUILD" -eq 0 ]]; then
    echo ""
    echo "[$(timestamp)] Step 2/3: Building rocrtst (${TARGET_DEVICE}) ..."

    if [[ ! -d "$ROCRTST_SRC" ]]; then
        echo "Error: rocrtst source not found at $ROCRTST_SRC" >&2
        exit 1
    fi

    # Build cmake prefix path.
    # Prefer the .venv SDK (has amd_smi, device-libs bitcode, LLVM, etc.)
    # over the system ROCm to ensure all dependencies are consistent.
    CMAKE_PFX="${ROCM_ROOT}"
    LLVM_ARGS=()
    NUMA_ARGS=()
    EXTRA_CMAKE_ARGS=()
    if [[ -n "${VENV_SDK:-}" && -d "${VENV_SDK:-}" ]]; then
        CMAKE_PFX="${VENV_SDK};${VENV_SDK}/lib/llvm;${CMAKE_PFX}"
        echo "  Using .venv SDK: $VENV_SDK"
        # Point LLVM to .venv SDK so bitcode is found relative to it
        if [[ -d "${VENV_SDK}/lib/cmake/llvm" ]]; then
            LLVM_ARGS+=(-DLLVM_DIR="${VENV_SDK}/lib/cmake/llvm")
        elif [[ -d "${VENV_SDK}/lib/llvm/lib/cmake/llvm" ]]; then
            LLVM_ARGS+=(-DLLVM_DIR="${VENV_SDK}/lib/llvm/lib/cmake/llvm")
        fi
        # Add rocm_sysdeps (numa) to the prefix path
        SYSDEPS_ROOT="${VENV_SDK}/lib/rocm_sysdeps"
        if [[ -d "$SYSDEPS_ROOT" ]]; then
            CMAKE_PFX="${SYSDEPS_ROOT};${CMAKE_PFX}"
            NUMA_ARGS+=(-DNUMA_DIR="${SYSDEPS_ROOT}/lib/cmake/NUMA")
        fi
        # Use the system hsa-runtime64 (newer than .venv SDK, has hsa_amd_signal_get_event_id)
        if [[ -d "${ROCM_ROOT}/lib/cmake/hsa-runtime64" ]]; then
            EXTRA_CMAKE_ARGS+=(-Dhsa-runtime64_DIR="${ROCM_ROOT}/lib/cmake/hsa-runtime64")
        fi
    else
        CMAKE_PFX="${ROCM_ROOT}/llvm;${CMAKE_PFX}"
        if [[ -d "${ROCM_ROOT}/llvm/lib/cmake/llvm" ]]; then
            LLVM_ARGS+=(-DLLVM_DIR="${ROCM_ROOT}/llvm/lib/cmake/llvm")
        fi
    fi

    # Use development HSA headers from the workspace (newer APIs than installed SDK).
    # rocrtst includes headers as "hsa/hsa_ext_amd.h" but the dev headers live
    # flat in inc/. Create a temporary shim directory with a "hsa" symlink.
    ROCR_INC="${PROJECT_DIR}/../../projects/rocr-runtime/runtime/hsa-runtime/inc"
    EXTRA_CXX_FLAGS=""
    if [[ -d "$ROCR_INC" ]]; then
        HSA_SHIM_DIR="${ROCRTST_BUILD_DIR}/_hsa_dev_headers"
        mkdir -p "$HSA_SHIM_DIR"
        ln -sfn "$(cd "$ROCR_INC" && pwd)" "$HSA_SHIM_DIR/hsa"
        EXTRA_CXX_FLAGS="-I${HSA_SHIM_DIR}"
        echo "  Using dev HSA headers: $ROCR_INC"
    fi
    # Add rocm_sysdeps headers (numa.h, etc.) if available
    SYSDEPS_INC="${VENV_SDK:-__none__}/lib/rocm_sysdeps/include"
    if [[ -d "$SYSDEPS_INC" ]]; then
        EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -I${SYSDEPS_INC}"
    fi
    # The bundled gtest is old and GTEST_IS_NULL_LITERAL_ breaks under C++17.
    # Define GTEST_ELLIPSIS_NEEDS_POD_ to disable the broken null-literal check.
    # -fpermissive works around type-mismatch bugs in old rocrtst assertions.
    EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -DGTEST_ELLIPSIS_NEEDS_POD_=1 -fpermissive"

    # Allow unresolved symbols at link time for dev-only APIs not yet in the
    # installed runtime (e.g. hsa_amd_agent_preload, hsa_amd_svm_discard_batch_async).
    # Tests calling these will segfault and be reported as failures in the report.
    EXTRA_LINK_FLAGS="-Wl,--unresolved-symbols=ignore-in-object-files -Wl,--allow-shlib-undefined"

    pushd "$ROCRTST_BUILD_DIR" > /dev/null
    cmake "$ROCRTST_SRC" \
        -DTARGET_DEVICES="${TARGET_DEVICE}" \
        -DCMAKE_PREFIX_PATH="${CMAKE_PFX}" \
        -DROCM_DIR="${ROCM_ROOT}" \
        "${LLVM_ARGS[@]}" \
        "${NUMA_ARGS[@]}" \
        "${EXTRA_CMAKE_ARGS[@]}" \
        -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${EXTRA_LINK_FLAGS}" \
        -DCMAKE_BUILD_TYPE=Release
    make -j"$JOBS"
    make rocrtst_kernels
    popd > /dev/null

    if [[ ! -f "$ROCRTST_BIN" ]]; then
        echo "Error: rocrtst64 not found at $ROCRTST_BIN" >&2
        exit 1
    fi
    echo "[$(timestamp)] rocrtst build complete."
else
    echo ""
    echo "[$(timestamp)] Step 2/3: Skipping rocrtst build (--skip-rocrtst-build)"
    if [[ ! -f "$ROCRTST_BIN" ]]; then
        echo "Error: rocrtst64 not found at $ROCRTST_BIN" >&2
        echo "  Build rocrtst first or remove --skip-rocrtst-build." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Run rocrtst under rocjitsu
# ---------------------------------------------------------------------------
echo ""
echo "[$(timestamp)] Step 3/3: Running rocrtst on simulated MI350X ..."
echo "  Config:  $(basename "$RJ_CONFIG")"
echo "  Schema:  $(basename "$RJ_SCHEMA")"
echo "  Binary:  $ROCRTST_BIN"
echo "  Device:  ${TARGET_DEVICE}"
if [[ -n "$GTEST_FILTER" ]]; then
    echo "  Filter:  $GTEST_FILTER"
fi
echo ""

# rocrtst discovers kernels relative to its working directory in the gfx950/ dir.
ROCRTST_DEVICE_DIR="${ROCRTST_BUILD_DIR}/${TARGET_DEVICE}"
ROCRTST_SYMLINK="${ROCRTST_DEVICE_DIR}/rocrtst64"
if [[ -L "$ROCRTST_SYMLINK" || -f "$ROCRTST_SYMLINK" ]]; then
    RUN_BIN="./rocrtst64"
    RUN_DIR="$ROCRTST_DEVICE_DIR"
else
    # Fall back to the main binary
    RUN_BIN="$ROCRTST_BIN"
    RUN_DIR="$ROCRTST_BUILD_DIR"
fi

# Build gtest arguments
GTEST_ARGS=(
    "--gtest_output=xml:${REPORT_XML}"
)
if [[ -n "$GTEST_FILTER" ]]; then
    GTEST_ARGS+=("--gtest_filter=${GTEST_FILTER}")
else
    # Exclude tests that dispatch GPU kernels (they hang on the simulator).
    # Override with --gtest-filter '*' to run everything.
    GTEST_ARGS+=("--gtest_filter=${DEFAULT_FILTER}")
    echo "  (Excluding kernel-dispatch tests; use --gtest-filter '*' to run all)"
fi
if [[ "$VERBOSE" -eq 1 ]]; then
    GTEST_ARGS+=("--gtest_print_time=1" "-v" "2")
fi

echo "  Timeout: ${TEST_TIMEOUT}s"

# Run under the rocjitsu interposer with a per-run timeout.
set +e
(
    cd "$RUN_DIR"
    timeout --signal=TERM --kill-after=10 "${TEST_TIMEOUT}" \
        env \
            LD_PRELOAD="$KMD_LIB" \
            LD_LIBRARY_PATH="${ROCM_ROOT}/lib:${ROCRTST_BUILD_DIR}:${LD_LIBRARY_PATH:-}" \
            RJ_CONFIG="$RJ_CONFIG" \
            RJ_SCHEMA="$RJ_SCHEMA" \
            HSA_ENABLE_SDMA=1 \
            ROCPROFILER_REGISTER_ENABLED=0 \
            "$RUN_BIN" "${GTEST_ARGS[@]}" 2>&1
) | tee "$REPORT_TXT"
TEST_EXIT=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
if [[ "$TEST_EXIT" -eq 124 ]]; then
    echo "[$(timestamp)] Test run TIMED OUT after ${TEST_TIMEOUT}s (exit code: $TEST_EXIT)"
else
    echo "[$(timestamp)] Test run finished (exit code: $TEST_EXIT)"
fi
echo "============================================================"
echo ""

# Parse summary from gtest output
PASSED=0; FAILED=0; SKIPPED=0; DISABLED=0; TOTAL=0
if [[ -f "$REPORT_TXT" ]]; then
    # Count completed test results
    PASSED=$(grep -cP '^\[\s+OK\s+\]' "$REPORT_TXT") || PASSED=0
    # Only count FAILED lines with timing info (avoid double-counting summary)
    FAILED=$(grep -cP '^\[\s+FAILED\s+\].*\(\d+ ms\)' "$REPORT_TXT") || FAILED=0
    SKIPPED=$(grep -cP '^\[\s+SKIPPED\s+\]' "$REPORT_TXT") || SKIPPED=0
    DISABLED=$(grep -oP 'YOU HAVE \K[0-9]+(?= DISABLED)' "$REPORT_TXT") || DISABLED=0
    # If gtest printed a summary, use its total; otherwise count RUN markers
    TOTAL=$(grep -oP '\[\s*=+\s*\]\s*\K[0-9]+(?=\s+tests?\s+(from|ran))' "$REPORT_TXT" | tail -1) || true
    if [[ -z "$TOTAL" || "$TOTAL" -eq 0 ]]; then
        RAN=$(grep -cP '^\[\s+RUN\s+\]' "$REPORT_TXT") || RAN=0
        TOTAL=$RAN
    fi
fi

# Summary
SUMMARY_FILE="${REPORT_DIR}/summary.txt"
{
    echo "rocrtst on rocjitsu MI350X - Test Report"
    echo "========================================="
    echo "Date:       $(timestamp)"
    echo "Device:     AMD Instinct MI350X (simulated, ${TARGET_DEVICE})"
    echo "Config:     $(basename "$RJ_CONFIG")"
    echo ""
    echo "Results"
    echo "-------"
    echo "  Total:    ${TOTAL}"
    echo "  Passed:   ${PASSED}"
    echo "  Failed:   ${FAILED}"
    echo "  Skipped:  ${SKIPPED}"
    echo "  Disabled: ${DISABLED}"
    echo "  Exit code: ${TEST_EXIT}"
    if [[ "$TEST_EXIT" -eq 124 ]]; then
        echo "  NOTE:     Test run was terminated after ${TEST_TIMEOUT}s timeout."
        echo "            Some tests hang on the simulated GPU (no kernel completion)."
    fi
    echo ""
    echo "Report files"
    echo "------------"
    echo "  Full log:  ${REPORT_TXT}"
    echo "  JUnit XML: ${REPORT_XML}"
    echo "  Summary:   ${SUMMARY_FILE}"

    if [[ "$FAILED" -gt 0 ]]; then
        echo ""
        echo "Failed tests"
        echo "------------"
        grep -P '^\[\s+FAILED\s+\]' "$REPORT_TXT" | sed 's/^/  /' || true
    fi
} | tee "$SUMMARY_FILE"

echo ""
echo "Reports written to: ${REPORT_DIR}/"

exit "$TEST_EXIT"
