// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/thread_trace/code_object.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/thread_trace/code_object_instrumentation.hpp"

#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstring>
#include <iterator>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
namespace code_object
{
namespace rocp_code_object = rocprofiler::code_object;

using set_type_t         = std::set<CodeobjCallbackRegistry*>;
using hsa_loader_table_t = hsa_ven_amd_loader_1_01_pfn_t;

struct CodeobjLoadInfo
{
    rocprofiler_agent_id_t agent      = {.handle = 0};
    uint64_t               id         = 0;
    uint64_t               load_addr  = 0;
    uint64_t               load_size  = 0;
    const void*            image      = nullptr;
    size_t                 image_size = 0;
};

struct LoadedMemoryCodeObject
{
    hsa_agent_t              agent              = {};
    hsa_loaded_code_object_t loaded_code_object = {};
    const void*              image              = nullptr;
    size_t                   image_size         = 0;
    int64_t                  load_delta         = 0;
};

struct InstrumentedUpload
{
    hsa_executable_t         executable              = {};
    hsa_code_object_reader_t reader                  = {};
    hsa_loaded_code_object_t loaded_code_object      = {};
    std::vector<uint8_t>     image                   = {};
    std::vector<uint64_t>    code_object_ids         = {};
    std::vector<uint64_t>    original_kernel_objects = {};
};

struct RegisteredKernelSymbol
{
    std::string name          = {};
    uint64_t    kernel_object = 0;
};

struct InstrumentedUploadResult
{
    InstrumentedUpload           upload = {};
    std::vector<CodeobjLoadInfo> infos  = {};
};

using upload_map_t = std::unordered_map<uint64_t, std::vector<InstrumentedUpload>>;

auto*
_static_registries()
{
    static auto*& reg = common::static_object<common::Synchronized<set_type_t>>::construct();
    return reg;
}

auto&
get_registries()
{
    return *CHECK_NOTNULL(_static_registries());
}

namespace
{
decltype(::hsa_executable_freeze)*&
get_freeze_function();

decltype(::hsa_executable_destroy)*&
get_destroy_function();
}  // namespace

hsa_loader_table_t&
get_loader_table()
{
    static auto _v = []() {
        auto _val = hsa_loader_table_t{};
        memset(&_val, 0, sizeof(hsa_loader_table_t));
        return _val;
    }();
    return _v;
}

std::mutex&
get_uploads_mutex()
{
    // HSA executable destruction can be called from ROCr/HIP teardown after C++ function-local
    // statics in this DSO have been destroyed. Keep this storage alive for the process lifetime.
    static auto* _v = new std::mutex{};
    return *CHECK_NOTNULL(_v);
}

upload_map_t&
get_uploads()
{
    // See get_uploads_mutex(). The uploaded executable copies must remain discoverable during
    // late hsa_executable_destroy callbacks.
    static auto* _v = new upload_map_t{};
    return *CHECK_NOTNULL(_v);
}

std::atomic<bool>&
get_finalized()
{
    static auto* _v = new std::atomic<bool>{false};
    return *CHECK_NOTNULL(_v);
}

bool&
get_upload_registration_active()
{
    thread_local auto _v = false;
    return _v;
}

struct UploadRegistrationGuard
{
    UploadRegistrationGuard()
    : previous(get_upload_registration_active())
    {
        get_upload_registration_active() = true;
    }

    ~UploadRegistrationGuard() { get_upload_registration_active() = previous; }

    bool previous = false;
};

bool
is_instrumented_executable(hsa_executable_t executable)
{
    auto lock = std::unique_lock{get_uploads_mutex()};
    for(const auto& [_, uploads] : get_uploads())
    {
        for(const auto& upload : uploads)
        {
            if(upload.executable.handle == executable.handle) return true;
        }
    }

    return false;
}

CodeobjLoadInfo
make_load_info(const rocprofiler::code_object::hsa::code_object& code_object)
{
    const auto& data = code_object.rocp_data;
    auto        info = CodeobjLoadInfo{data.rocp_agent,
                                data.code_object_id,
                                static_cast<uint64_t>(data.load_delta),
                                data.load_size};

    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY && data.memory_base != 0 &&
       data.memory_size != 0)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        info.image      = reinterpret_cast<const void*>(data.memory_base);
        info.image_size = data.memory_size;
    }

    return info;
}

std::vector<CodeobjLoadInfo>
collect_load_infos(hsa_executable_t executable)
{
    auto infos = std::vector<CodeobjLoadInfo>{};

    rocp_code_object::iterate_loaded_code_objects(
        [&](const rocp_code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable.handle != executable.handle) return;
            infos.emplace_back(make_load_info(code_object));
        });

    return infos;
}

std::optional<LoadedMemoryCodeObject>
get_loaded_memory_code_object(hsa_loaded_code_object_t loaded_code_object)
{
    if(get_loader_table().hsa_ven_amd_loader_loaded_code_object_get_info == nullptr)
        return std::nullopt;

    auto get_info = [&](hsa_ven_amd_loader_loaded_code_object_info_t attribute, auto* value) {
        return get_loader_table().hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object, attribute, value);
    };

    auto storage_type = uint32_t{};
    if(get_info(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
                &storage_type) != HSA_STATUS_SUCCESS)
        return std::nullopt;

    if(storage_type != HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) return std::nullopt;

    auto memory_base = uint64_t{};
    auto memory_size = uint64_t{};
    auto info        = LoadedMemoryCodeObject{.loaded_code_object = loaded_code_object};

    if(get_info(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &info.agent) !=
       HSA_STATUS_SUCCESS)
        return std::nullopt;
    if(get_info(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                &memory_base) != HSA_STATUS_SUCCESS)
        return std::nullopt;
    if(get_info(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                &memory_size) != HSA_STATUS_SUCCESS)
        return std::nullopt;
    if(get_info(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA, &info.load_delta) !=
       HSA_STATUS_SUCCESS)
        return std::nullopt;

    if(memory_base == 0 || memory_size == 0) return std::nullopt;

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    info.image      = reinterpret_cast<const void*>(memory_base);
    info.image_size = static_cast<size_t>(memory_size);

    return info;
}

hsa_status_t
collect_loaded_memory_code_object(hsa_executable_t,
                                  hsa_loaded_code_object_t loaded_code_object,
                                  void*                    data)
{
    auto* infos = static_cast<std::vector<LoadedMemoryCodeObject>*>(data);
    if(auto info = get_loaded_memory_code_object(loaded_code_object)) infos->emplace_back(*info);

    return HSA_STATUS_SUCCESS;
}

std::vector<LoadedMemoryCodeObject>
collect_memory_loaded_code_objects(hsa_executable_t executable)
{
    auto infos = std::vector<LoadedMemoryCodeObject>{};

    if(get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects != nullptr)
        get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
            executable, collect_loaded_memory_code_object, &infos);

    return infos;
}

std::vector<RegisteredKernelSymbol>
get_registered_kernel_symbols(hsa_executable_t         executable,
                              hsa_loaded_code_object_t loaded_code_object)
{
    auto symbols = std::vector<RegisteredKernelSymbol>{};

    rocp_code_object::iterate_loaded_code_objects(
        [&](const rocp_code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable.handle != executable.handle ||
               code_object.hsa_code_object.handle != loaded_code_object.handle)
                return;

            for(const auto& symbol : code_object.symbols)
            {
                if(!symbol || symbol->rocp_data.kernel_object == 0 ||
                   symbol->rocp_data.kernel_name == nullptr)
                    continue;

                symbols.emplace_back(RegisteredKernelSymbol{symbol->rocp_data.kernel_name,
                                                            symbol->rocp_data.kernel_object});
            }
        });

    return symbols;
}

bool
register_kernel_object_replacements(hsa_executable_t         source_executable,
                                    hsa_loaded_code_object_t source_loaded_code_object,
                                    InstrumentedUpload&      upload)
{
    auto originals    = get_registered_kernel_symbols(source_executable, source_loaded_code_object);
    auto replacements = get_registered_kernel_symbols(upload.executable, upload.loaded_code_object);

    if(originals.empty() || replacements.empty()) return false;

    auto replacement_by_name = std::unordered_map<std::string, std::vector<uint64_t>>{};
    for(const auto& replacement : replacements)
        replacement_by_name[replacement.name].emplace_back(replacement.kernel_object);

    auto replacement_index_by_name = std::unordered_map<std::string, size_t>{};
    upload.original_kernel_objects.reserve(originals.size());
    for(const auto& original : originals)
    {
        auto replacement_itr = replacement_by_name.find(original.name);
        if(replacement_itr == replacement_by_name.end()) continue;

        auto& index = replacement_index_by_name[original.name];
        if(index >= replacement_itr->second.size()) continue;

        const auto replacement_kernel_object = replacement_itr->second.at(index++);
        rocp_code_object::add_kernel_object_replacement(original.kernel_object,
                                                        replacement_kernel_object);
        upload.original_kernel_objects.emplace_back(original.kernel_object);
    }

    return upload.original_kernel_objects.size() == originals.size();
}

bool
has_registries()
{
    return get_registries().rlock([](const set_type_t& t) { return !t.empty(); });
}

void
notify_load(const std::vector<CodeobjLoadInfo>& infos)
{
    if(infos.empty()) return;

    get_registries().wlock([&](set_type_t& t) {
        for(const auto& info : infos)
        {
            for(auto* reg : t)
                reg->ld_fn(info.agent, info.id, info.load_addr, info.load_size);
        }
    });
}

void
notify_unload(const std::vector<uint64_t>& ids)
{
    if(ids.empty()) return;

    get_registries().wlock([&](set_type_t& t) {
        for(auto id : ids)
        {
            for(auto* reg : t)
                reg->unld_fn(id);
        }
    });
}

void
destroy_unregistered_upload(InstrumentedUpload& upload)
{
    auto* core = CHECK_NOTNULL(rocprofiler::hsa::get_core_table());

    if(upload.executable.handle != 0 && core->hsa_executable_destroy_fn != nullptr)
        core->hsa_executable_destroy_fn(upload.executable);

    if(upload.reader.handle != 0 && core->hsa_code_object_reader_destroy_fn != nullptr)
        core->hsa_code_object_reader_destroy_fn(upload.reader);
}

void
destroy_registered_upload(InstrumentedUpload& upload)
{
    auto* core = CHECK_NOTNULL(rocprofiler::hsa::get_core_table());

    for(auto original_kernel_object : upload.original_kernel_objects)
        rocp_code_object::remove_kernel_object_replacement(original_kernel_object);

    if(upload.executable.handle != 0) CHECK_NOTNULL(get_destroy_function())(upload.executable);

    if(upload.reader.handle != 0 && core->hsa_code_object_reader_destroy_fn != nullptr)
        core->hsa_code_object_reader_destroy_fn(upload.reader);
}

std::pair<hsa_profile_t, hsa_default_float_rounding_mode_t>
get_executable_create_options(hsa_executable_t executable)
{
    auto* core = CHECK_NOTNULL(rocprofiler::hsa::get_core_table());

    auto profile       = HSA_PROFILE_FULL;
    auto rounding_mode = HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;

    if(core->hsa_executable_get_info_fn != nullptr)
    {
        core->hsa_executable_get_info_fn(executable, HSA_EXECUTABLE_INFO_PROFILE, &profile);
        core->hsa_executable_get_info_fn(
            executable, HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &rounding_mode);
    }

    return {profile, rounding_mode};
}

std::optional<InstrumentedUploadResult>
create_instrumented_upload(hsa_executable_t source_executable, const LoadedMemoryCodeObject& info)
{
    if(info.image == nullptr || info.image_size == 0) return std::nullopt;
    if(!contains_s_ttracedata(info.image, info.image_size)) return std::nullopt;

    auto* core = CHECK_NOTNULL(rocprofiler::hsa::get_core_table());
    if(core->hsa_executable_create_alt_fn == nullptr ||
       core->hsa_code_object_reader_create_from_memory_fn == nullptr ||
       core->hsa_executable_load_agent_code_object_fn == nullptr ||
       core->hsa_executable_freeze_fn == nullptr ||
       get_loader_table().hsa_ven_amd_loader_loaded_code_object_get_info == nullptr)
    {
        return std::nullopt;
    }

    auto  result = InstrumentedUploadResult{};
    auto& upload = result.upload;

    upload.image.resize(info.image_size);
    std::memcpy(upload.image.data(), info.image, info.image_size);
    if(!instrument_codeobj_image(upload.image, static_cast<uint64_t>(info.load_delta)))
        return std::nullopt;

    auto [profile, rounding_mode] = get_executable_create_options(source_executable);

    auto status =
        core->hsa_executable_create_alt_fn(profile, rounding_mode, nullptr, &upload.executable);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to create instrumented code object executable: " << status;
        return std::nullopt;
    }

    status = core->hsa_code_object_reader_create_from_memory_fn(
        upload.image.data(), upload.image.size(), &upload.reader);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to create instrumented code object reader: " << status;
        destroy_unregistered_upload(upload);
        return std::nullopt;
    }

    status = core->hsa_executable_load_agent_code_object_fn(
        upload.executable, info.agent, upload.reader, nullptr, &upload.loaded_code_object);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to load instrumented code object: " << status;
        destroy_unregistered_upload(upload);
        return std::nullopt;
    }

    status = core->hsa_executable_freeze_fn(upload.executable, nullptr);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to freeze instrumented code object executable: " << status;
        destroy_unregistered_upload(upload);
        return std::nullopt;
    }

    {
        auto guard = UploadRegistrationGuard{};
        status     = rocp_code_object::executable_freeze_internal(upload.executable);
        if(status != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << "Failed to register instrumented code object: " << status;
            destroy_unregistered_upload(upload);
            return std::nullopt;
        }
    }

    result.infos = collect_load_infos(upload.executable);
    if(result.infos.empty())
    {
        destroy_registered_upload(upload);
        return std::nullopt;
    }

    if(!register_kernel_object_replacements(source_executable, info.loaded_code_object, upload))
    {
        ROCP_WARNING << "Failed to register instrumented kernel object replacements";
        destroy_registered_upload(upload);
        return std::nullopt;
    }

    upload.code_object_ids.reserve(result.infos.size());
    for(const auto& load_info : result.infos)
        upload.code_object_ids.emplace_back(load_info.id);

    return result;
}

std::vector<CodeobjLoadInfo>
upload_instrumented_code_objects(hsa_executable_t                           executable,
                                 const std::vector<LoadedMemoryCodeObject>& infos)
{
    {
        auto lock = std::unique_lock{get_uploads_mutex()};
        if(get_uploads().find(executable.handle) != get_uploads().end()) return {};
    }

    auto uploads   = std::vector<InstrumentedUpload>{};
    auto new_infos = std::vector<CodeobjLoadInfo>{};

    for(const auto& info : infos)
    {
        auto result = create_instrumented_upload(executable, info);
        if(!result) continue;

        new_infos.insert(new_infos.end(), result->infos.begin(), result->infos.end());
        uploads.emplace_back(std::move(result->upload));
    }

    if(!uploads.empty())
    {
        auto  lock = std::unique_lock{get_uploads_mutex()};
        auto& slot = get_uploads()[executable.handle];
        slot.insert(slot.end(),
                    std::make_move_iterator(uploads.begin()),
                    std::make_move_iterator(uploads.end()));
    }

    return new_infos;
}

std::vector<CodeobjLoadInfo>
instrument_executable(hsa_executable_t executable)
{
    if(get_finalized().load(std::memory_order_acquire)) return {};

    if(get_upload_registration_active() || !has_registries() ||
       is_instrumented_executable(executable))
        return {};

    return upload_instrumented_code_objects(executable,
                                            collect_memory_loaded_code_objects(executable));
}

std::vector<uint64_t>
take_instrumented_upload_ids(hsa_executable_t executable, std::vector<InstrumentedUpload>& uploads)
{
    {
        auto lock = std::unique_lock{get_uploads_mutex()};
        if(auto it = get_uploads().find(executable.handle); it != get_uploads().end())
        {
            uploads = std::move(it->second);
            get_uploads().erase(it);
        }
    }

    auto ids = std::vector<uint64_t>{};
    for(const auto& upload : uploads)
        ids.insert(ids.end(), upload.code_object_ids.begin(), upload.code_object_ids.end());

    return ids;
}

CodeobjCallbackRegistry::CodeobjCallbackRegistry(LoadCallback _ld, UnloadCallback _unld)
: ld_fn(std::move(_ld))
, unld_fn(std::move(_unld))
{
    get_registries().wlock([this](set_type_t& t) { t.insert(this); });
}

CodeobjCallbackRegistry::~CodeobjCallbackRegistry()
{
    if(auto* reg = _static_registries()) reg->wlock([this](set_type_t& t) { t.erase(this); });
}

void
CodeobjCallbackRegistry::IterateLoaded() const
{
    auto executables = std::unordered_map<uint64_t, hsa_executable_t>{};

    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            if(!is_instrumented_executable(code_object.hsa_executable))
                executables[code_object.hsa_executable.handle] = code_object.hsa_executable;
        });

    for(const auto& [_, executable] : executables)
        instrument_executable(executable);

    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            auto info = make_load_info(code_object);
            ld_fn(info.agent, info.id, info.load_addr, info.load_size);
        });
}

namespace
{
decltype(::hsa_executable_freeze)*&
get_freeze_function()
{
    static decltype(::hsa_executable_freeze)* _v = nullptr;
    return _v;
}

decltype(::hsa_executable_destroy)*&
get_destroy_function()
{
    static decltype(::hsa_executable_destroy)* _v = nullptr;
    return _v;
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    // Call underlying function
    hsa_status_t status = CHECK_NOTNULL(get_freeze_function())(executable, options);
    if(status != HSA_STATUS_SUCCESS) return status;

    if(get_finalized().load(std::memory_order_acquire)) return HSA_STATUS_SUCCESS;

    if(!has_registries()) return HSA_STATUS_SUCCESS;

    notify_load(collect_load_infos(executable));

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    if(get_finalized().load(std::memory_order_acquire))
        return CHECK_NOTNULL(get_destroy_function())(executable);

    auto ids = std::vector<uint64_t>{};
    for(const auto& info : collect_load_infos(executable))
        ids.emplace_back(info.id);

    auto uploads          = std::vector<InstrumentedUpload>{};
    auto instrumented_ids = take_instrumented_upload_ids(executable, uploads);
    ids.insert(ids.end(), instrumented_ids.begin(), instrumented_ids.end());

    notify_unload(ids);

    for(auto& upload : uploads)
        destroy_registered_upload(upload);

    // Call underlying function
    return CHECK_NOTNULL(get_destroy_function())(executable);
}
}  // namespace

void
initialize(HsaApiTable* table)
{
    get_finalized().store(false, std::memory_order_release);

    auto& core_table = *table->core_;

    if(core_table.hsa_system_get_major_extension_table_fn != nullptr)
    {
        auto status = core_table.hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_loader_table_t), &get_loader_table());
        ROCP_WARNING_IF(status != HSA_STATUS_SUCCESS)
            << "hsa_system_get_major_extension_table for AMD loader failed: " << status;
    }

    get_freeze_function()                = CHECK_NOTNULL(core_table.hsa_executable_freeze_fn);
    get_destroy_function()               = CHECK_NOTNULL(core_table.hsa_executable_destroy_fn);
    core_table.hsa_executable_freeze_fn  = executable_freeze;
    core_table.hsa_executable_destroy_fn = executable_destroy;

    static auto callback_registered = false;
    if(!callback_registered)
    {
        rocp_code_object::add_executable_freeze_callback(
            [](hsa_executable_t executable) { notify_load(instrument_executable(executable)); });
        callback_registered = true;
    }

    LOG_IF(FATAL, get_freeze_function() == core_table.hsa_executable_freeze_fn)
        << "infinite recursion";
    LOG_IF(FATAL, get_destroy_function() == core_table.hsa_executable_destroy_fn)
        << "infinite recursion";
}

void
finalize()
{
    get_finalized().store(true, std::memory_order_release);
    get_registries().wlock([](set_type_t& t) { t.clear(); });
}

}  // namespace code_object
}  // namespace thread_trace
}  // namespace rocprofiler
