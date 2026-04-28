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
#   --timeout            SEC   total test run timeout    (default: 600)
#   --per-test-timeout   SEC   per-test timeout for kernel-dispatch tests (default: 120)
#   --verbose                  verbose test output
#   -h, --help                 show this help

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROCRTST_SRC="${PROJECT_DIR}/../../projects/rocr-runtime/rocrtst/suites/test_common"
VENV_DIR="${PROJECT_DIR}/.venv"
ROCM_PIP_INDEX="https://repo.amd.com/rocm/whl/gfx950-dcgpu/"

# ---------------------------------------------------------------------------
# Set up .venv with ROCm SDK if not already present
# ---------------------------------------------------------------------------
setup_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        # Verify the SDK is actually installed
        local sdk_dir
        sdk_dir="$(find "$VENV_DIR" -maxdepth 5 -name '_rocm_sdk_devel' -type d 2>/dev/null | head -1)"
        if [[ -n "$sdk_dir" && -d "$sdk_dir" ]]; then
            return 0  # Already set up
        fi
        echo "  .venv exists but ROCm SDK is missing — reinstalling packages ..."
    else
        echo "  Creating .venv ..."
        python3 -m venv "$VENV_DIR"
    fi

    echo "  Installing ROCm SDK packages (this may take a few minutes) ..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install --index-url "$ROCM_PIP_INDEX" \
        "rocm[libraries,devel]" --quiet
    echo "  .venv setup complete."
}

setup_venv

# Locate the .venv SDK (has hsa-runtime64, amd_smi, LLVM, device-libs, etc.)
# This is the sole ROCm SDK used for building and running — no system ROCm.
VENV_SDK="$(find "$VENV_DIR" -maxdepth 5 -name '_rocm_sdk_devel' -type d 2>/dev/null | head -1)"
if [[ -z "${VENV_SDK:-}" || ! -d "${VENV_SDK:-}" ]]; then
    echo "Error: ROCm SDK not found in .venv after setup. Check pip install output." >&2
    exit 1
fi
ROCM_ROOT="${VENV_SDK}"

# Defaults
RJ_BUILD_DIR="${PROJECT_DIR}/build"
ROCRTST_BUILD_DIR="${PROJECT_DIR}/build/rocrtst"
REPORT_DIR="${PROJECT_DIR}/build/rocrtst-report"
GTEST_FILTER=""
SKIP_RJ_BUILD=0
SKIP_ROCRTST_BUILD=0
JOBS="$(nproc)"
VERBOSE=0
TEST_TIMEOUT=600
PER_TEST_TIMEOUT=120

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
        --per-test-timeout)   PER_TEST_TIMEOUT="$2"; shift 2 ;;
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

# Tests unconditionally excluded:
#   - Agent_Preload_Latency: calls hsa_amd_agent_preload which is unresolved (crashes).
#   - IPC: requires fork + shared memory between processes (not yet supported by interposer).
ALWAYS_EXCLUDED="\
rocrtstPerf.Agent_Preload_Latency:\
rocrtstFunc.IPC"

# Tests that dispatch GPU kernels. These work correctly but run slowly under
# ISA simulation (~40s per dispatch). They are run individually with a
# per-test timeout so one slow test doesn't block the entire suite.
KERNEL_DISPATCH_TESTS=(
    rocrtst.Test_Example
    rocrtst.Test_Example_InterruptDisabled
    rocrtst.Test_MetadataPrefetchPacket
    rocrtstFunc.MemoryAccessTests
    rocrtstFunc.MemoryAccessCoherent
    rocrtstFunc.GroupMemoryAllocationTest
    rocrtstFunc.GpuCoreDump_DefaultPattern
    rocrtstFunc.GpuCoreDump_CustomPattern
    rocrtstFunc.GpuCoreDump_DisableFlag
    rocrtstFunc.GpuCoreDump_PatternSubstitution
    rocrtstFunc.GpuCoreDump_InvalidPath
    rocrtstFunc.GpuCoreDump_ContentIntegrity
    rocrtstFunc.GpuCoreDump_PipePattern
    rocrtstFunc.Memory_Atomic_Add_Test
    rocrtstFunc.Memory_Atomic_Sub_Test
    rocrtstFunc.Memory_Atomic_And_Test
    rocrtstFunc.Memory_Atomic_Or_Test
    rocrtstFunc.Memory_Atomic_Xor_Test
    rocrtstFunc.Memory_Atomic_Min_Test
    rocrtstFunc.Memory_Atomic_Max_Test
    rocrtstFunc.Memory_Atomic_Inc_Test
    rocrtstFunc.Memory_Atomic_Dec_Test
    rocrtstFunc.Memory_Atomic_Xchg_Test
    rocrtstFunc.SvmMemory_Basic_Test
    rocrtstFunc.VirtMemory_Access_Test
    rocrtstFunc.VirtMemory_Aliasing_Test
    rocrtstFunc.Counted_Queue_Dispatch_Test
    rocrtstFunc.Counted_Queue_Multithreaded_Dispatch_Test
    rocrtstFunc.Counted_Queue_Overflow_And_Wraparound_Test
    rocrtstNeg.Queue_Validation_InvalidDimension
    rocrtstNeg.Queue_Validation_InvalidGroupMemory
    rocrtstNeg.Queue_Validation_InvalidKernelObject
    rocrtstNeg.Queue_Validation_InvalidPacket
    rocrtstPerf.Memory_Async_Copy
    rocrtstPerf.Memory_Async_Copy_On_Engine
    rocrtstPerf.ENQUEUE_LATENCY
    rocrtstPerf.AQL_Dispatch_Time_Single_SpinWait
    rocrtstPerf.AQL_Dispatch_Time_Single_Interrupt
    rocrtstPerf.AQL_Dispatch_Time_Multi_SpinWait
    rocrtstPerf.AQL_Dispatch_Time_Multi_Interrupt
)

# Build the negative filter string for the fast (API-only) pass
KERNEL_FILTER_NEG=$(IFS=:; echo "${KERNEL_DISPATCH_TESTS[*]}")
DEFAULT_FILTER="-${ALWAYS_EXCLUDED}:${KERNEL_FILTER_NEG}"

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

    # Build cmake prefix path — use the .venv SDK exclusively.
    SYSDEPS_ROOT="${VENV_SDK}/lib/rocm_sysdeps"
    CMAKE_PFX="${VENV_SDK};${VENV_SDK}/lib/llvm;${SYSDEPS_ROOT}"
    echo "  Using .venv SDK: $VENV_SDK"

    LLVM_ARGS=()
    NUMA_ARGS=()
    EXTRA_CMAKE_ARGS=()

    # LLVM (for kernel compilation)
    if [[ -d "${VENV_SDK}/lib/cmake/llvm" ]]; then
        LLVM_ARGS+=(-DLLVM_DIR="${VENV_SDK}/lib/cmake/llvm")
    elif [[ -d "${VENV_SDK}/lib/llvm/lib/cmake/llvm" ]]; then
        LLVM_ARGS+=(-DLLVM_DIR="${VENV_SDK}/lib/llvm/lib/cmake/llvm")
    fi

    # NUMA from rocm_sysdeps
    if [[ -d "$SYSDEPS_ROOT" ]]; then
        NUMA_ARGS+=(-DNUMA_DIR="${SYSDEPS_ROOT}/lib/cmake/NUMA")
    fi

    # hsa-runtime64 from .venv SDK
    if [[ -d "${VENV_SDK}/lib/cmake/hsa-runtime64" ]]; then
        EXTRA_CMAKE_ARGS+=(-Dhsa-runtime64_DIR="${VENV_SDK}/lib/cmake/hsa-runtime64")
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
    # Add rocm_sysdeps headers (numa.h, etc.)
    SYSDEPS_INC="${VENV_SDK}/lib/rocm_sysdeps/include"
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
        -DROCM_DIR="${VENV_SDK}" \
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

# Common environment for running under the rocjitsu interposer
RUN_ENV=(
    LD_PRELOAD="$KMD_LIB"
    LD_LIBRARY_PATH="${VENV_SDK}/lib:${VENV_SDK}/lib/rocm_sysdeps/lib:${ROCRTST_BUILD_DIR}:${LD_LIBRARY_PATH:-}"
    RJ_CONFIG="$RJ_CONFIG"
    RJ_SCHEMA="$RJ_SCHEMA"
    HSA_ENABLE_SDMA=1
    ROCPROFILER_REGISTER_ENABLED=0
)

VERBOSE_ARGS=()
if [[ "$VERBOSE" -eq 1 ]]; then
    VERBOSE_ARGS=("--gtest_print_time=1" "-v" "2")
fi

if [[ -n "$GTEST_FILTER" ]]; then
    # User specified an explicit filter — run in a single pass.
    echo "  Filter:  $GTEST_FILTER"
    echo "  Timeout: ${TEST_TIMEOUT}s"
    set +e
    (
        cd "$RUN_DIR"
        timeout --signal=TERM --kill-after=10 "${TEST_TIMEOUT}" \
            env "${RUN_ENV[@]}" \
            "$RUN_BIN" --gtest_output=xml:"${REPORT_XML}" \
                        --gtest_filter="${GTEST_FILTER}" \
                        "${VERBOSE_ARGS[@]}" 2>&1
    ) | tee "$REPORT_TXT"
    TEST_EXIT=${PIPESTATUS[0]}
    set -e
else
    # Two-phase run:
    #   Phase 1: Fast API-only tests (no kernel dispatch) — run as a batch.
    #   Phase 2: Kernel-dispatch tests — run individually with per-test timeout
    #            because ISA simulation is slow (~40s per kernel dispatch).
    echo "  Phase 1: API-only tests (batch, timeout ${TEST_TIMEOUT}s)"
    echo "  Phase 2: Kernel-dispatch tests (individual, timeout ${PER_TEST_TIMEOUT}s each)"
    echo ""

    # --- Phase 1: API-only tests ---
    echo "--- Phase 1: API-only tests ---"
    set +e
    (
        cd "$RUN_DIR"
        timeout --signal=TERM --kill-after=10 "${TEST_TIMEOUT}" \
            env "${RUN_ENV[@]}" \
            "$RUN_BIN" --gtest_output=xml:"${REPORT_XML}" \
                        --gtest_filter="${DEFAULT_FILTER}" \
                        "${VERBOSE_ARGS[@]}" 2>&1
    ) | tee "$REPORT_TXT"
    PHASE1_EXIT=${PIPESTATUS[0]}
    set -e
    echo ""
    echo "--- Phase 1 complete (exit $PHASE1_EXIT) ---"
    echo ""

    # --- Phase 2: Kernel-dispatch tests (run each with per-test timeout) ---
    echo "--- Phase 2: Kernel-dispatch tests (${#KERNEL_DISPATCH_TESTS[@]} tests, ${PER_TEST_TIMEOUT}s each) ---"
    PHASE2_PASS=0
    PHASE2_FAIL=0
    PHASE2_TIMEOUT=0
    for TEST_NAME in "${KERNEL_DISPATCH_TESTS[@]}"; do
        printf "  %-55s " "$TEST_NAME"
        set +e
        TEST_OUT=$(
            cd "$RUN_DIR"
            timeout --signal=TERM --kill-after=10 "${PER_TEST_TIMEOUT}" \
                env "${RUN_ENV[@]}" \
                "$RUN_BIN" --gtest_filter="${TEST_NAME}" \
                            "${VERBOSE_ARGS[@]}" 2>&1
        )
        RC=$?
        set -e

        # Append to the text report log
        echo "$TEST_OUT" >> "$REPORT_TXT"

        if [[ $RC -eq 124 || $RC -eq 137 ]]; then
            echo "TIMEOUT (${PER_TEST_TIMEOUT}s)"
            ((PHASE2_TIMEOUT++)) || true
        elif echo "$TEST_OUT" | grep -qP '^\[\s+OK\s+\]'; then
            ELAPSED=$(echo "$TEST_OUT" | grep -oP '\(\K\d+(?= ms\))' | tail -1)
            echo "PASSED (${ELAPSED:-?} ms)"
            ((PHASE2_PASS++)) || true
        else
            echo "FAILED (exit $RC)"
            ((PHASE2_FAIL++)) || true
        fi
    done
    echo ""
    echo "--- Phase 2 complete: ${PHASE2_PASS} passed, ${PHASE2_FAIL} failed, ${PHASE2_TIMEOUT} timed out ---"

    # Overall exit code: non-zero if any phase failed
    if [[ $PHASE1_EXIT -ne 0 || $PHASE2_FAIL -gt 0 ]]; then
        TEST_EXIT=1
    else
        TEST_EXIT=0
    fi
fi

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
