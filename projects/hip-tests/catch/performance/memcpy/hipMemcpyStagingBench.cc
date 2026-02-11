/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "memcpy_performance_common.hh"

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 *
 * Benchmark suite targeting hipMemcpy H2D staging path optimizations:
 */
class MemcpyStagingBench : public Benchmark<MemcpyStagingBench> {
 public:
  void operator()(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipMemcpy(dst, src, size, kind)); }
  }
};

static void RunBench(LinearAllocs dst_alloc, LinearAllocs src_alloc,
                     size_t size, hipMemcpyKind kind) {
  MemcpyStagingBench bench;
  bench.AddSectionName(std::to_string(size / 1024) + " KB");
  bench.AddSectionName(GetAllocationSectionName(src_alloc));
  bench.AddSectionName(GetAllocationSectionName(dst_alloc));

  LinearAllocGuard<int> src(src_alloc, size);
  LinearAllocGuard<int> dst(dst_alloc, size);
  bench.Run(dst.ptr(), src.ptr(), size, kind);
}

TEST_CASE("Perf_H2D_Pageable") {
  const auto size = GENERATE(256, 1_KB, 4_KB, 16_KB, 64_KB, 256_KB,
                             1_MB, 2_MB, 4_MB);
  RunBench(LinearAllocs::hipMalloc, LinearAllocs::malloc,
           size, hipMemcpyHostToDevice);
}

/**
 * End doxygen group memcpy.
 * @}
 */
