// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L1VectorCache::ensure_line(uint64_t addr) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");

  uint8_t line_buf[LINE_SIZE];
  l2_->fetch_line(line_addr, line_buf);
  cache_.fill_line(addr, line_buf);
}

void L1VectorCache::read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype,
                               bool non_temporal) {
  if (mtype == Mtype::UC || non_temporal) {
    l2_->read(addr, dst, size, mtype);
    return;
  }

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

    // Functional execution can schedule dependent dispatches on different CUs or
    // XCDs without modeling global cache probes. Stores are write-through below,
    // so backing memory is authoritative; force each vector load to refill through
    // L2 instead of reusing a clean line that may predate another CU's store.
    cache_.invalidate(ea);

    ensure_line(ea);
    cache_.read_line(ea, dst + copied, line_offset, chunk);
    copied += chunk;
  }
}

void L1VectorCache::write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                                bool non_temporal) {
  util::Logger::vm([&](auto &os) {
    if (addr >= 0x4d00c00000ULL && addr < 0x4d00c00100ULL) {
      uint32_t val = 0;
      if (size >= 4)
        std::memcpy(&val, src, 4);
      else if (size >= 2)
        std::memcpy(&val, src, size);
      else
        val = src[0];
      static thread_local uint32_t tw = 0;
      if (++tw <= 20)
        os << std::format("L1_WRITE @{:#x} size={} val={:#x} mtype={}", addr, size, val,
                          static_cast<int>(mtype));
    }
  });
  if (mtype == Mtype::UC || non_temporal) {
    l2_->write(addr, src, size, mtype);
    return;
  }

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

    ensure_line(ea);
    cache_.write_line(ea, src + copied, line_offset, chunk);

    // Write through to L2 for all cacheable stores. This ensures partial writes
    // from different CUs sharing the same L2 are properly merged at byte
    // granularity via L2::write(), rather than full-line replacement via
    // writeback_line() during L1 eviction/flush.
    l2_->write(ea, src + copied, chunk, mtype);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag);
    assert(tag != nullptr && "ensure_line must guarantee hit");

    // L1 line stays clean since L2 has the authoritative copy.
    tag->coherence =
        (mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED : simdojo::CoherenceState::EXCLUSIVE;
    tag->dirty = false;
    copied += chunk;
  }
}

void L1VectorCache::load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                         uint32_t num_elems, uint8_t *dst, Mtype mtype, bool non_temporal) {
  uint32_t stride = num_elems * elem_size;
  // Iterate only over lanes that are active according to lane_mask.
  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t lane = std::countr_zero(remaining);
    remaining &= remaining - 1; // clear lowest set bit
    uint64_t base = addrs[lane];
    for (uint32_t e = 0; e < num_elems; ++e) {
      uint64_t ea = base + e * elem_size;
      read_bytes(ea, dst + lane * stride + e * elem_size, elem_size, mtype, non_temporal);
    }
  }
}

void L1VectorCache::store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                          uint32_t num_elems, const uint8_t *src, Mtype mtype, bool non_temporal) {
  uint32_t stride = num_elems * elem_size;
  uint32_t active_lanes = std::popcount(lane_mask);
  ++store_count_;
  if (active_lanes > 0)
    ++store_active_count_;
  store_l2_writes_ += active_lanes * num_elems;
  // Iterate only over lanes that are active according to lane_mask.
  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t lane = std::countr_zero(remaining);
    remaining &= remaining - 1; // clear lowest set bit
    uint64_t base = addrs[lane];
    for (uint32_t e = 0; e < num_elems; ++e) {
      uint64_t ea = base + e * elem_size;
      write_bytes(ea, src + lane * stride + e * elem_size, elem_size, mtype, non_temporal);
    }
  }
}

void L1VectorCache::flush_all() {
  // All stores are written through to L2 during execution, so no dirty lines
  // to write back. Just invalidate.
  cache_.invalidate_all();
}

} // namespace amdgpu
} // namespace rocjitsu
