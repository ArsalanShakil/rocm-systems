/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "pinned-host-memory-pool.h"

#include "context.h"
#include "hip.h"
#include "sys.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <new>
#include <syslog.h>

namespace hipFile {

size_t
PinnedHostMemoryPool::normalizeSize(size_t size)
{
    size_t normalized_size = std::max(size, sizeof(void *));

    if ((normalized_size & (normalized_size - 1)) == 0) {
        return normalized_size;
    }

    if (normalized_size > (std::numeric_limits<size_t>::max() >> 1)) {
        return normalized_size;
    }

    size_t power_of_two = sizeof(void *);
    while (power_of_two < normalized_size) {
        power_of_two <<= 1;
    }

    return power_of_two;
}

PinnedHostMemoryPool::Bucket *
PinnedHostMemoryPool::findBucket(size_t size) noexcept
{
    for (Bucket &bucket : buckets) {
        if (bucket.size == size) {
            return &bucket;
        }
    }
    return nullptr;
}

PinnedHostMemoryPool::Bucket *
PinnedHostMemoryPool::findOrCreateBucket(size_t size)
{
    if (Bucket *bucket = findBucket(size)) {
        return bucket;
    }

    buckets.push_back(Bucket{size, nullptr});
    return &buckets.back();
}

void *
PinnedHostMemoryPool::allocate(size_t size)
{
    const size_t normalized_size = normalizeSize(size);

    {
        std::lock_guard<std::mutex> lock{mutex};
        if (Bucket *bucket = findBucket(normalized_size); bucket && bucket->free_list) {
            void *ptr           = bucket->free_list;
            bucket->free_list = *static_cast<void **>(ptr);
            std::fprintf(stderr, "PinnedHostMemoryPool::allocate requested=%zu returned=%p\n", size, ptr);
            return ptr;
        }
    }

    void *ptr = Context<Hip>::get()->hipHostMalloc(normalized_size, 0);
    std::fprintf(stderr, "PinnedHostMemoryPool::allocate requested=%zu returned=%p\n", size, ptr);
    return ptr;
}

void
PinnedHostMemoryPool::release(void *ptr, size_t size) noexcept
{
    if (!ptr) {
        return;
    }

    const size_t normalized_size = normalizeSize(size);

    try {
        std::lock_guard<std::mutex> lock{mutex};
        Bucket                     *bucket = findOrCreateBucket(normalized_size);
        *static_cast<void **>(ptr)        = bucket->free_list;
        bucket->free_list                 = ptr;
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Unable to return pinned host allocation to pool; leaking.");
    }
}

}
