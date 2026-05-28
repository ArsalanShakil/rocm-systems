/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>

#include "alloc.h"
#include "comm.h"
#include "mem_manager.h"

namespace RcclUnitTesting
{

// Diagnostic dump invoked right before each real HIP allocation in
// MemManagerRealMem.*. Kept after the RUN_ISOLATED_TEST -> TEST_F refactor so
// that any future GPU OOM / runtime reject (HIP bug, noisy neighbour, leaked
// previous run) still surfaces inline with the gtest failure line.
inline void logHipDiag(const char* where)
{
    int        dev    = -1;
    hipError_t devErr = hipGetDevice(&dev);

    int        devCount = -1;
    hipError_t cntErr   = hipGetDeviceCount(&devCount);

    size_t     freeMem  = 0;
    size_t     totalMem = 0;
    hipError_t infoErr  = hipMemGetInfo(&freeMem, &totalMem);

    int rtVer  = 0;
    int drvVer = 0;
    (void)hipRuntimeGetVersion(&rtVer);
    (void)hipDriverGetVersion(&drvVer);

    hipDeviceProp_t prop    = {};
    hipError_t      propErr = (dev >= 0) ? hipGetDeviceProperties(&prop, dev)
                                         : hipErrorInvalidValue;

    auto envOrUnset = [](const char* name) -> const char* {
        const char* v = getenv(name);
        return v ? v : "<unset>";
    };

    rlimit asLim   = {};
    rlimit lockLim = {};
    getrlimit(RLIMIT_AS, &asLim);
    getrlimit(RLIMIT_MEMLOCK, &lockLim);

    char buf[2048];
    int  n = snprintf(buf, sizeof(buf),
            "[ MEM DIAG ] %s\n"
            "             pid=%d hip_rt=%d hip_drv=%d devCount=%d(err=%d) "
            "dev=%d(err=%d) arch=%s(err=%d)\n"
            "             hipMemGetInfo: err=%d free=%zu (%.2f GiB) "
            "total=%zu (%.2f GiB)\n"
            "             rlimit: AS cur=%zu max=%zu MEMLOCK cur=%zu max=%zu\n"
            "             env: HIP_VISIBLE_DEVICES=%s ROCR_VISIBLE_DEVICES=%s "
            "AMD_VISIBLE_DEVICES=%s\n"
            "             env: HSA_NO_SCRATCH_RECLAIM=%s HSA_ENABLE_SDMA=%s "
            "GPU_MAX_HEAP_SIZE=%s GPU_MAX_ALLOC_PERCENT=%s\n",
            where,
            static_cast<int>(getpid()), rtVer, drvVer,
            devCount, static_cast<int>(cntErr),
            dev, static_cast<int>(devErr),
            (propErr == hipSuccess ? prop.gcnArchName : "?"),
            static_cast<int>(propErr),
            static_cast<int>(infoErr),
            freeMem, freeMem / (1024.0 * 1024.0 * 1024.0),
            totalMem, totalMem / (1024.0 * 1024.0 * 1024.0),
            static_cast<size_t>(asLim.rlim_cur),
            static_cast<size_t>(asLim.rlim_max),
            static_cast<size_t>(lockLim.rlim_cur),
            static_cast<size_t>(lockLim.rlim_max),
            envOrUnset("HIP_VISIBLE_DEVICES"),
            envOrUnset("ROCR_VISIBLE_DEVICES"),
            envOrUnset("AMD_VISIBLE_DEVICES"),
            envOrUnset("HSA_NO_SCRATCH_RECLAIM"),
            envOrUnset("HSA_ENABLE_SDMA"),
            envOrUnset("GPU_MAX_HEAP_SIZE"),
            envOrUnset("GPU_MAX_ALLOC_PERCENT"));
    (void)n;

    fputs(buf, stdout);
    fflush(stdout);
    fputs(buf, stderr);
    fflush(stderr);

    // Also drop a per-process file so the diag survives even when CI pipes
    // truncate or interleave stdout/stderr. Picked up via `cat /tmp/rccl_mem_diag_*`.
    char path[128];
    snprintf(path, sizeof(path), "/tmp/rccl_mem_diag_%d.log", static_cast<int>(getpid()));
    if(FILE* fp = fopen(path, "a")) {
        fputs(buf, fp);
        fclose(fp);
    }
}

// RAII wrappers for resources allocated inside MemManager tests. Each guard
// runs its release call from the destructor so a mid-test ASSERT_* still
// leaves the process clean. Ownership transfer to the manager is done via the
// explicit release() / dismiss() methods.

// Closes a file descriptor on destruction unless dismissed.
class ScopedFd
{
public:
    ScopedFd() = default;

    explicit ScopedFd(int fd) : fd_(fd) {}

    ~ScopedFd()
    {
        if(fd_ >= 0) ::close(fd_);
    }

    ScopedFd(const ScopedFd&)            = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    ScopedFd(ScopedFd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

    ScopedFd& operator=(ScopedFd&& o) noexcept
    {
        if(this != &o)
        {
            if(fd_ >= 0) ::close(fd_);
            fd_   = o.fd_;
            o.fd_ = -1;
        }
        return *this;
    }

    int  get() const { return fd_; }
    int  release() { int f = fd_; fd_ = -1; return f; }
    bool valid() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

// hipMalloc / hipFree.
class HipDeviceBuffer
{
public:
    HipDeviceBuffer() = default;

    explicit HipDeviceBuffer(size_t size)
    {
        if(hipMalloc(&ptr_, size) != hipSuccess) ptr_ = nullptr;
    }

    ~HipDeviceBuffer()
    {
        if(ptr_) (void)hipFree(ptr_);
    }

    HipDeviceBuffer(const HipDeviceBuffer&)            = delete;
    HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;

    HipDeviceBuffer(HipDeviceBuffer&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }

    HipDeviceBuffer& operator=(HipDeviceBuffer&& o) noexcept
    {
        if(this != &o)
        {
            if(ptr_) (void)hipFree(ptr_);
            ptr_   = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }

    void* get() const { return ptr_; }
    void* release() { void* p = ptr_; ptr_ = nullptr; return p; }

private:
    void* ptr_ = nullptr;
};

// ncclCudaHostCalloc / ncclCudaHostFree. Used when the manager later takes
// ownership of the host backup pointer — call release() before transferring.
template<typename T>
class HipHostBuffer
{
public:
    HipHostBuffer() = default;

    explicit HipHostBuffer(size_t nelem)
    {
        if(ncclCudaHostCalloc(&ptr_, nelem) != ncclSuccess) ptr_ = nullptr;
    }

    ~HipHostBuffer()
    {
        if(ptr_) (void)ncclCudaHostFree(ptr_);
    }

    HipHostBuffer(const HipHostBuffer&)            = delete;
    HipHostBuffer& operator=(const HipHostBuffer&) = delete;

    HipHostBuffer(HipHostBuffer&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }

    HipHostBuffer& operator=(HipHostBuffer&& o) noexcept
    {
        if(this != &o)
        {
            if(ptr_) (void)ncclCudaHostFree(ptr_);
            ptr_   = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    T* release() { T* p = ptr_; ptr_ = nullptr; return p; }

private:
    T* ptr_ = nullptr;
};

// hipStreamCreateWithFlags / hipStreamDestroy.
class HipStream
{
public:
    HipStream()
    {
        if(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess)
        {
            stream_ = nullptr;
        }
    }

    ~HipStream()
    {
        if(stream_) (void)hipStreamDestroy(stream_);
    }

    HipStream(const HipStream&)            = delete;
    HipStream& operator=(const HipStream&) = delete;

    hipStream_t get() const { return stream_; }
    bool        valid() const { return stream_ != nullptr; }

private:
    hipStream_t stream_ = nullptr;
};

// VMM allocation triple (hipMemCreate + reserve + map). Teardown in the
// HIP-required order on destruction (hipMemUnmap -> hipMemRelease ->
// hipMemAddressFree). After a successful allocateVmmPosixFd() the struct
// owns the resources; the manager only stores bookkeeping, not the mapping.
struct VmmPosixAllocation
{
    void*                           ptr    = nullptr;
    hipDeviceptr_t                  pdev   = 0;
    size_t                          size   = 0;
    hipMemGenericAllocationHandle_t handle = 0;
    bool                            owned  = false;

    VmmPosixAllocation() = default;

    ~VmmPosixAllocation()
    {
        if(!owned) return;
        (void)hipMemUnmap(pdev, size);
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(pdev, size);
    }

    VmmPosixAllocation(const VmmPosixAllocation&)            = delete;
    VmmPosixAllocation& operator=(const VmmPosixAllocation&) = delete;

    VmmPosixAllocation(VmmPosixAllocation&& o) noexcept
        : ptr(o.ptr), pdev(o.pdev), size(o.size), handle(o.handle), owned(o.owned)
    {
        o.owned = false;
    }

    VmmPosixAllocation& operator=(VmmPosixAllocation&& o) noexcept
    {
        if(this != &o)
        {
            if(owned)
            {
                (void)hipMemUnmap(pdev, size);
                (void)hipMemRelease(handle);
                (void)hipMemAddressFree(pdev, size);
            }
            ptr     = o.ptr;
            pdev    = o.pdev;
            size    = o.size;
            handle  = o.handle;
            owned   = o.owned;
            o.owned = false;
        }
        return *this;
    }

    void dismiss() { owned = false; }
};

// Allocates a chunk of device memory via the HIP VMM API with a POSIX-fd
// shareable handle. Mirrors the prop layout of ncclCuMemAlloc:
//   - Pinned + Device location
//   - requestedHandleType = POSIX_FILE_DESCRIPTOR
//   - allocFlags.gpuDirectRDMACapable = 1 (ROCM-2550 workaround; without it
//     hipMemMap can SIGSEGV on AMD)
// `requestedSize` is rounded up to the minimum granularity reported by HIP.
// On success `out->owned` is set to true so the destructor releases everything.
inline void allocateVmmPosixFd(int dev, size_t requestedSize, VmmPosixAllocation* out)
{
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(hipSetDevice(dev), hipSuccess);

    logHipDiag("allocateVmmPosixFd: before hipMemCreate(POSIX_FD)");

    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = hipMemLocationTypeDevice;
    prop.location.id                     = dev;
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    ASSERT_EQ(hipMemGetAllocationGranularity(&granularity, &prop,
                                             hipMemAllocationGranularityMinimum),
              hipSuccess);
    ASSERT_GT(granularity, 0u);
    size_t size = ((requestedSize + granularity - 1) / granularity) * granularity;

    hipMemGenericAllocationHandle_t handle = 0;
    ASSERT_EQ(hipMemCreate(&handle, size, &prop, 0), hipSuccess);

    hipDeviceptr_t pdev = 0;
    ASSERT_EQ(hipMemAddressReserve(&pdev, size, granularity, 0, 0), hipSuccess);

    if(hipMemMap(pdev, size, 0, handle, 0) != hipSuccess)
    {
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(pdev, size);
        FAIL() << "hipMemMap failed";
    }

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
    if(hipMemSetAccess(pdev, size, &accessDesc, 1) != hipSuccess)
    {
        (void)hipMemUnmap(pdev, size);
        (void)hipMemRelease(handle);
        (void)hipMemAddressFree(pdev, size);
        FAIL() << "hipMemSetAccess failed";
    }

    out->ptr    = reinterpret_cast<void*>(pdev);
    out->pdev   = pdev;
    out->size   = size;
    out->handle = handle;
    out->owned  = true;
}

// Allocates via ncclCuMemAlloc (the RCCL wrapper). On success the caller-side
// release path is `ncclCuMemFree(ptr, manager=nullptr)`. We model that as
// owned=true + manual ncclCuMemFree-on-destruction by using a small dedicated
// guard class below instead of overloading VmmPosixAllocation.
struct NcclCuMemRawAllocation
{
    void*                           ptr    = nullptr;
    hipMemGenericAllocationHandle_t handle = 0;
    size_t                          size   = 0;
    bool                            owned  = false;

    NcclCuMemRawAllocation() = default;
    ~NcclCuMemRawAllocation()
    {
        if(!owned) return;
        (void)ncclCuMemFree(ptr, /*manager=*/nullptr);
    }
    NcclCuMemRawAllocation(const NcclCuMemRawAllocation&)            = delete;
    NcclCuMemRawAllocation& operator=(const NcclCuMemRawAllocation&) = delete;

    void dismiss() { owned = false; }
};

inline void allocateViaNcclCuMemAlloc(int dev, size_t requestedSize,
                                      NcclCuMemRawAllocation* out)
{
    ASSERT_NE(out, nullptr);
    ASSERT_EQ(hipSetDevice(dev), hipSuccess);

    void*                           ptr    = nullptr;
    hipMemGenericAllocationHandle_t handle = 0;
    ncclResult_t r = ncclCuMemAlloc(&ptr, &handle, hipMemHandleTypePosixFileDescriptor,
                                    requestedSize, /*manager=*/nullptr);
    ASSERT_EQ(r, ncclSuccess);
    ASSERT_NE(ptr, nullptr);
    ASSERT_NE(reinterpret_cast<void*>(handle), nullptr);

    out->ptr    = ptr;
    out->handle = handle;
    out->size   = requestedSize;
    out->owned  = true;
}

// ncclCuMemAlloc + ncclCuMemFree(ptr, manager). The buffer auto-frees with the
// supplied manager (so the manager's bookkeeping stays consistent). Tests that
// want to drive ncclCuMemFree manually (to assert on internal state) can call
// release() to take ownership back.
class NcclCuMemBuffer
{
public:
    NcclCuMemBuffer() = default;

    NcclCuMemBuffer(size_t                 size,
                    struct ncclMemManager* manager,
                    ncclMemType_t          memType)
        : manager_(manager)
    {
        if(ncclCuMemAlloc(&ptr_, &handle_, hipMemHandleTypePosixFileDescriptor,
                          size, manager, memType)
           != ncclSuccess)
        {
            ptr_ = nullptr;
        }
    }

    ~NcclCuMemBuffer()
    {
        if(ptr_) (void)ncclCuMemFree(ptr_, manager_);
    }

    NcclCuMemBuffer(const NcclCuMemBuffer&)            = delete;
    NcclCuMemBuffer& operator=(const NcclCuMemBuffer&) = delete;

    NcclCuMemBuffer(NcclCuMemBuffer&& o) noexcept
        : ptr_(o.ptr_), handle_(o.handle_), manager_(o.manager_)
    {
        o.ptr_     = nullptr;
        o.handle_  = 0;
        o.manager_ = nullptr;
    }

    NcclCuMemBuffer& operator=(NcclCuMemBuffer&& o) noexcept
    {
        if(this != &o)
        {
            if(ptr_) (void)ncclCuMemFree(ptr_, manager_);
            ptr_       = o.ptr_;
            handle_    = o.handle_;
            manager_   = o.manager_;
            o.ptr_     = nullptr;
            o.handle_  = 0;
            o.manager_ = nullptr;
        }
        return *this;
    }

    void*                           get() const { return ptr_; }
    hipMemGenericAllocationHandle_t handle() const { return handle_; }
    void*                           release()
    {
        void* p = ptr_;
        ptr_    = nullptr;
        return p;
    }

private:
    void*                           ptr_     = nullptr;
    hipMemGenericAllocationHandle_t handle_  = 0;
    struct ncclMemManager*          manager_ = nullptr;
};

// ncclCuda{Calloc,Malloc,CallocAsync} + ncclCudaFree(ptr, manager). T is the
// allocation element type (int, char, uint64_t, etc.). Construct as empty
// and bind via assign() once the underlying API call has succeeded.
template<typename T>
class NcclCudaBuffer
{
public:
    NcclCudaBuffer() = default;

    NcclCudaBuffer(T* ptr, struct ncclMemManager* manager) : ptr_(ptr), manager_(manager) {}

    ~NcclCudaBuffer()
    {
        if(ptr_) (void)ncclCudaFree(ptr_, manager_);
    }

    NcclCudaBuffer(const NcclCudaBuffer&)            = delete;
    NcclCudaBuffer& operator=(const NcclCudaBuffer&) = delete;

    NcclCudaBuffer(NcclCudaBuffer&& o) noexcept : ptr_(o.ptr_), manager_(o.manager_)
    {
        o.ptr_     = nullptr;
        o.manager_ = nullptr;
    }

    NcclCudaBuffer& operator=(NcclCudaBuffer&& o) noexcept
    {
        if(this != &o)
        {
            if(ptr_) (void)ncclCudaFree(ptr_, manager_);
            ptr_       = o.ptr_;
            manager_   = o.manager_;
            o.ptr_     = nullptr;
            o.manager_ = nullptr;
        }
        return *this;
    }

    void assign(T* ptr, struct ncclMemManager* manager)
    {
        ptr_     = ptr;
        manager_ = manager;
    }

    T* get() const { return ptr_; }
    T* release()
    {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

private:
    T*                     ptr_     = nullptr;
    struct ncclMemManager* manager_ = nullptr;
};

// Common fixture for tests that exercise real HIP / VMM allocations against a
// freshly-initialised ncclMemManager. SetUp pins device 0, allocates a fresh
// ncclComm and calls ncclMemManagerInit; TearDown destroys the manager (if a
// test didn't already do it explicitly) and deletes the comm. Both run via
// gtest infrastructure even when an ASSERT_* fires inside the test body, so
// each test body is responsible only for the test-specific resources, which
// must use the RAII wrappers above.
class MemManagerRealMemFixture : public ::testing::Test
{
protected:
    ncclComm* comm = nullptr;

    void SetUp() override
    {
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        comm          = new ncclComm();
        comm->cudaDev = 0;
        ASSERT_EQ(ncclMemManagerInit(comm), ncclSuccess);
    }

    void TearDown() override
    {
        if(comm)
        {
            if(comm->memManager) (void)ncclMemManagerDestroy(comm);
            delete comm;
            comm = nullptr;
        }
    }
};

} // namespace RcclUnitTesting
