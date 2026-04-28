/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

namespace hipFile {

class PinnedHostMemoryPool {
public:
    void *allocate(size_t size);
    void  release(void *ptr, size_t size) noexcept;

private:
    struct Bucket {
        size_t size;
        void  *free_list;
    };

    static size_t normalizeSize(size_t size);
    Bucket       *findBucket(size_t size) noexcept;
    Bucket       *findOrCreateBucket(size_t size);

    std::mutex          mutex;
    std::vector<Bucket> buckets;
};

}
