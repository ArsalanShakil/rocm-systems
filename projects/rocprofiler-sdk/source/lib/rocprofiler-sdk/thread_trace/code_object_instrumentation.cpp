// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/thread_trace/code_object_instrumentation.hpp"

#include <amd_comgr/amd_comgr.h>

#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

namespace rocprofiler
{
namespace thread_trace
{
namespace code_object
{
namespace
{
constexpr uint32_t s_ttracedata_encoding = 0xbf960000;
constexpr uint32_t s_nop_0_encoding      = 0b101111111u << 23;

struct symbol_info
{
    uint64_t vaddr    = 0;
    uint64_t faddr    = 0;
    uint64_t mem_size = 0;
    bool     kernel   = false;
};

struct comgr_data
{
    amd_comgr_data_t handle{};
    ~comgr_data()
    {
        if(handle.handle != 0) amd_comgr_release_data(handle);
    }
};

uint32_t
read_u32(const std::vector<uint8_t>& data, size_t offset)
{
    auto value = uint32_t{};
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

void
write_u32(std::vector<uint8_t>& data, size_t offset, uint32_t value)
{
    std::memcpy(data.data() + offset, &value, sizeof(value));
}

bool
is_s_mov_b64_literal(uint32_t inst)
{
    constexpr uint32_t s_mov_b64_literal_mask  = 0xff80ffff;
    constexpr uint32_t s_mov_b64_literal_value = 0xbe8001ff;
    return (inst & s_mov_b64_literal_mask) == s_mov_b64_literal_value;
}

bool
is_s_and_b64(uint32_t inst)
{
    constexpr uint32_t sop2_encoding = 2;
    constexpr uint32_t s_and_b64_op  = 13;

    const auto op  = (inst >> 23) & 0x7f;
    const auto enc = (inst >> 30) & 0x3;

    return enc == sop2_encoding && op == s_and_b64_op;
}

bool
find_s_ttracedata(const uint8_t* data, size_t begin, size_t end, size_t step)
{
    if(data == nullptr || step == 0 || end < begin + sizeof(uint32_t)) return false;

    for(size_t i = begin; i + sizeof(uint32_t) <= end; i += step)
    {
        auto inst = uint32_t{};
        std::memcpy(&inst, data + i, sizeof(inst));
        if(inst == s_ttracedata_encoding) return true;
    }

    return false;
}

amd_comgr_status_t
symbol_callback(amd_comgr_symbol_t symbol, void* userdata)
{
    auto* ctx = static_cast<std::pair<amd_comgr_data_t, std::vector<symbol_info>>*>(userdata);

    auto type   = amd_comgr_symbol_type_t{};
    auto status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE, &type);
    if(status != AMD_COMGR_STATUS_SUCCESS) return status;
    if(type != AMD_COMGR_SYMBOL_TYPE_FUNC && type != AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL)
        return AMD_COMGR_STATUS_SUCCESS;

    auto vaddr    = uint64_t{};
    auto mem_size = uint64_t{};
    status        = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_VALUE, &vaddr);
    if(status != AMD_COMGR_STATUS_SUCCESS) return status;

    status = amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_SIZE, &mem_size);
    if(status != AMD_COMGR_STATUS_SUCCESS) return status;

    auto faddr  = uint64_t{};
    auto slice  = uint64_t{};
    auto nobits = false;
    status      = amd_comgr_map_elf_virtual_address_to_code_object_offset(
        ctx->first, vaddr, &faddr, &slice, &nobits);
    if(status == AMD_COMGR_STATUS_SUCCESS && !nobits)
        ctx->second.push_back(
            {vaddr, faddr, mem_size, type == AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL});

    return AMD_COMGR_STATUS_SUCCESS;
}

std::vector<symbol_info>
get_codeobj_symbols(const void* image, size_t image_size)
{
    if(image == nullptr || image_size == 0) return {};

    auto data = comgr_data{};
    if(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data.handle) !=
       AMD_COMGR_STATUS_SUCCESS)
        return {};

    if(amd_comgr_set_data(data.handle, image_size, static_cast<const char*>(image)) !=
       AMD_COMGR_STATUS_SUCCESS)
        return {};

    auto ctx = std::pair<amd_comgr_data_t, std::vector<symbol_info>>{data.handle, {}};
    if(amd_comgr_iterate_symbols(data.handle, symbol_callback, &ctx) != AMD_COMGR_STATUS_SUCCESS)
        return {};

    auto&      symbols = ctx.second;
    const auto has_kernel_symbols =
        std::any_of(symbols.begin(), symbols.end(), [](const auto& sym) { return sym.kernel; });
    if(has_kernel_symbols)
    {
        symbols.erase(
            std::remove_if(
                symbols.begin(), symbols.end(), [](const auto& sym) { return !sym.kernel; }),
            symbols.end());
    }

    std::sort(symbols.begin(), symbols.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.faddr, lhs.vaddr) < std::tie(rhs.faddr, rhs.vaddr);
    });

    symbols.erase(std::unique(symbols.begin(),
                              symbols.end(),
                              [](const auto& lhs, const auto& rhs) {
                                  return lhs.faddr == rhs.faddr && lhs.vaddr == rhs.vaddr;
                              }),
                  symbols.end());

    return symbols;
}

}  // namespace

bool
contains_s_ttracedata(const void* data, size_t size)
{
    return find_s_ttracedata(static_cast<const uint8_t*>(data), 0, size, 1);
}

bool
patch_kernel_code(std::vector<uint8_t>& image, size_t begin, size_t end, uint64_t new_kernel_addr)
{
    if(begin >= image.size()) return false;
    end = std::min(end, image.size());
    if(end < begin + sizeof(uint32_t)) return false;

    const auto low_addr = static_cast<uint32_t>(new_kernel_addr & 0xffffff);
    const auto hi_addr  = static_cast<uint32_t>((new_kernel_addr >> 24) & 0xffffff);

    auto found_movs     = size_t{0};
    auto patched_and    = false;
    auto patched_ttrace = false;

    for(size_t offset = begin; offset + sizeof(uint32_t) <= end; offset += sizeof(uint32_t))
    {
        const auto inst = read_u32(image, offset);

        if(inst == s_ttracedata_encoding)
        {
            write_u32(image, offset, s_nop_0_encoding);
            patched_ttrace = true;
            continue;
        }

        if(found_movs < 2 && is_s_mov_b64_literal(inst))
        {
            if(offset + 2 * sizeof(uint32_t) > end) return false;

            write_u32(image, offset + sizeof(uint32_t), (found_movs == 0) ? low_addr : hi_addr);
            ++found_movs;
            offset += sizeof(uint32_t);
            continue;
        }

        if(found_movs >= 2 && !patched_and && is_s_and_b64(inst))
        {
            write_u32(image, offset, s_nop_0_encoding);
            patched_and = true;
        }
    }

    return found_movs == 2 && patched_and && patched_ttrace;
}

bool
instrument_codeobj_image(std::vector<uint8_t>& image, uint64_t load_addr)
{
    if(!contains_s_ttracedata(image.data(), image.size())) return false;

    auto symbols = get_codeobj_symbols(image.data(), image.size());
    if(symbols.empty()) return false;

    auto traced_kernel_count = size_t{0};
    for(size_t i = 0; i < symbols.size(); ++i)
    {
        const auto& symbol = symbols.at(i);
        auto        end    = image.size();

        if(symbol.mem_size != 0)
        {
            end = static_cast<size_t>(
                std::min<uint64_t>(image.size(), symbol.faddr + symbol.mem_size));
        }
        else if(i + 1 < symbols.size() && symbols.at(i + 1).faddr > symbol.faddr)
        {
            end = static_cast<size_t>(std::min<uint64_t>(image.size(), symbols.at(i + 1).faddr));
        }

        if(symbol.faddr >= end) continue;

        if(!find_s_ttracedata(
               image.data(), static_cast<size_t>(symbol.faddr), end, sizeof(uint32_t)))
            continue;

        ++traced_kernel_count;

        const auto new_kernel_addr = load_addr + symbol.vaddr;
        if(!patch_kernel_code(image, static_cast<size_t>(symbol.faddr), end, new_kernel_addr))
            return false;
    }

    return traced_kernel_count > 0;
}

std::vector<uint64_t>
get_kernel_symbol_vaddrs(const void* image, size_t image_size)
{
    auto symbols = get_codeobj_symbols(image, image_size);
    auto vaddrs  = std::vector<uint64_t>{};
    vaddrs.reserve(symbols.size());

    for(const auto& symbol : symbols)
        vaddrs.emplace_back(symbol.vaddr);

    return vaddrs;
}

}  // namespace code_object
}  // namespace thread_trace
}  // namespace rocprofiler
