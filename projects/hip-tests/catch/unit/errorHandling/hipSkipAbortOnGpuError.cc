/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipSkipAbortOnGpuError HIP_SKIP_ABORT_ON_GPU_ERROR
 * @{
 * @ingroup ErrorHandlingTest
 *
 * Tests that GPU hardware exceptions return a HIP error code instead of
 * calling abort(). HIP_SKIP_ABORT_ON_GPU_ERROR defaults to 1 in CLR so
 * these tests run unconditionally on AMD hardware, mirroring CUDA's
 * CU_COREDUMP_SKIP_ABORT behavior.
 *
 * Covered HSA status codes (AMD-only):
 *   HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION  -- CUDA_EXCEPTION_4/8
 *   HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION -- CUDA_EXCEPTION_5/6/7
 *
 * HSA_STATUS_ERROR_EXCEPTION (device-side assert) is covered by
 * unit/assertion/assert.cc. HSA_STATUS_ERROR_OUT_OF_REGISTERS ("Kernel has
 * requested more VGPRs than are available on this agent") is a dispatch-time
 * check; the compiler caps VGPR usage via register spilling, so this status
 * cannot be reached from normal user-space HIP code and is not tested here.
 *
 * NOTE: Each test case must be run in a separate process invocation (which
 * ctest does by default). After a GPU hardware exception, HIP's global GPU
 * error state is set, causing subsequent HIP API calls in the same process
 * to fail. Running via ctest ensures per-test process isolation.
 *
 * Reference:
 *   https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-error-reporting
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#if defined(__linux__)
#include <signal.h>
#endif

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Triggers HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION via __builtin_trap(), which
// emits an illegal GPU instruction opcode (analogous to CUDA_EXCEPTION_4:
// Warp Illegal Instruction, and CUDA_EXCEPTION_8: Warp Invalid PC).
__global__ void illegal_instruction_kernel() {
  __builtin_trap();
}

// Triggers HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION by storing to an
// address in the LDS aperture range through a flat (global) pointer.
// The LDS aperture on AMD GCN/RDNA starts at 0xFFFF000000000000.
// Analogous to CUDA_EXCEPTION_5/6/7: Warp Out-of-range / Misaligned /
// Invalid Address Space.
__global__ void aperture_violation_kernel() {
  volatile int* lds_aperture = reinterpret_cast<volatile int*>(0xFFFF000000000000ULL);
  *lds_aperture = 1;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/**
 * Test Description
 * ------------------------
 *  - Launches a kernel that executes an illegal GPU instruction via
 *    __builtin_trap() (HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION).
 *  - Expects hipErrorLaunchFailure from hipStreamSynchronize() instead of
 *    abort(), relying on the default HIP_SKIP_ABORT_ON_GPU_ERROR=1 behavior.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipSkipAbortOnGpuError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_HipSkipAbortOnGpuError_IllegalInstruction) {
#if HT_AMD
  // ROCr may attempt to spawn a GPU coredump child process on exception.
  // If the coredump binary is absent (e.g. in CI), execvp fails and the
  // broken pipe raises SIGPIPE. Ignore it so the error propagates normally.
#if defined(__linux__)
  signal(SIGPIPE, SIG_IGN);
#endif
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  illegal_instruction_kernel<<<1, 1, 0, stream>>>();
  HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorLaunchFailure);
#else
  HIP_SKIP_TEST("AMD-only: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION behavior.");
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Launches a kernel that writes to an address in the LDS aperture range
 *    through a flat pointer, triggering an aperture violation
 *    (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION).
 *  - Expects hipErrorIllegalAddress from hipStreamSynchronize() instead of
 *    abort(), relying on the default HIP_SKIP_ABORT_ON_GPU_ERROR=1 behavior.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipSkipAbortOnGpuError.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_HipSkipAbortOnGpuError_ApertureViolation) {
#if HT_AMD
#if defined(__linux__)
  signal(SIGPIPE, SIG_IGN);
#endif
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  aperture_violation_kernel<<<1, 1, 0, stream>>>();
  HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorIllegalAddress);
#else
  HIP_SKIP_TEST("AMD-only: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION behavior.");
#endif
}

/**
 * End doxygen group ErrorHandlingTest.
 * @}
 */
