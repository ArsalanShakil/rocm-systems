// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbt_hooks.cpp
/// @brief HSA tools load-time DBT hook for translating AMDGPU code objects.
///
/// @details ROCR loads this shared library through `HSA_TOOLS_LIB` during
/// `hsa_init()`. The hook saves the API table entries that are present at
/// `OnLoad()` time, installs wrappers for code-object reader creation/destruction
/// and agent code-object loads, and invokes rocjitsu DBT before ROCR sees a
/// memory-backed guest code object. The MVP is deliberately strict: when
/// translation is requested and fails, the hook returns an HSA error instead of
/// retrying the original reader, because the original ELF may target a different
/// GPU ISA.

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "util/arena_alloc.h"
#include "util/intrusive_list.h"
#include "util/log.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <execinfo.h>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <signal.h>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using rocjitsu::AmdGpuCodeObject;
using rocjitsu::arch_for_elf_mach;
using rocjitsu::BinaryTranslator;
using rocjitsu::BinaryTranslatorOptions;
using rocjitsu::DiagnosticKind;
using rocjitsu::DiagnosticSeverity;
using rocjitsu::EF_AMDGPU_MACH;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1100;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942;
using rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950;
using rocjitsu::Elf64_Ehdr;
using rocjitsu::elf_mach_for_arch;
using rocjitsu::elf_mach_name;
using rocjitsu::ELFCLASS64;
using rocjitsu::EM_AMDGPU;
using rocjitsu::has_error_diagnostic;
using rocjitsu::TranslationDiagnostic;

enum HookLogLevel : int {
  kLogDisabled = 0,
  kLogInfo = 1,
  kLogVerbose = 2,
  kLogDebug = 3,
};

std::atomic<int> g_log_level{kLogDisabled};
std::atomic<bool> g_signal_backtrace_enabled{false};

constexpr uint32_t kHsaAmdAgentInfoDriverNodeId = 0xA004;
constexpr hsa_amd_memory_pool_info_t kHsaAmdMemoryPoolInfoSegment = 0;
constexpr hsa_amd_memory_pool_info_t kHsaAmdMemoryPoolInfoGlobalFlags = 1;
constexpr hsa_amd_memory_pool_info_t kHsaAmdMemoryPoolInfoRuntimeAllocAllowed = 5;
constexpr hsa_amd_memory_pool_info_t kHsaAmdMemoryPoolInfoLocation = 17;

struct TargetInfo {
  std::string_view name;
  rj_code_arch_t arch;
  uint32_t mach;
};

constexpr std::array<uint32_t, 5> kAcceptedConcreteTargetMachs = {
    EF_AMDGPU_MACH_AMDGCN_GFX942, EF_AMDGPU_MACH_AMDGCN_GFX950, EF_AMDGPU_MACH_AMDGCN_GFX1100,
    EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201};

constexpr std::array<TargetInfo, 4> kArchAliases = {{
    {"cdna3", ROCJITSU_CODE_ARCH_CDNA3, elf_mach_for_arch(ROCJITSU_CODE_ARCH_CDNA3)},
    {"cdna4", ROCJITSU_CODE_ARCH_CDNA4, elf_mach_for_arch(ROCJITSU_CODE_ARCH_CDNA4)},
    {"rdna3", ROCJITSU_CODE_ARCH_RDNA3, elf_mach_for_arch(ROCJITSU_CODE_ARCH_RDNA3)},
    {"rdna4", ROCJITSU_CODE_ARCH_RDNA4, elf_mach_for_arch(ROCJITSU_CODE_ARCH_RDNA4)},
}};

#define RJ_AMD_EXT_HAS_FIELD(table, field)                                                         \
  ((table) != nullptr &&                                                                           \
   (table)->version.minor_id >= offsetof(AmdExtTable, field) + sizeof((table)->field))

struct HookConfig {
  TargetInfo target{};
  std::optional<TargetInfo> source_override;
  std::optional<TargetInfo> guest_target;
  uint32_t host_gpu_id = 0;
  int log_level = kLogDisabled;
  bool signal_backtrace = false;
};

[[nodiscard]] std::optional<TargetInfo> parse_target(std::string_view value) {
  for (uint32_t mach : kAcceptedConcreteTargetMachs) {
    std::string_view name = elf_mach_name(mach);
    if (value == name) {
      const rj_code_arch_t arch = arch_for_elf_mach(mach);
      if (arch != ROCJITSU_CODE_ARCH_INVALID)
        return TargetInfo{name, arch, mach};
    }
  }
  for (const TargetInfo &target : kArchAliases) {
    if (value == target.name)
      return target;
  }
  return std::nullopt;
}

[[nodiscard]] int clamp_log_level(int value) {
  if (value < kLogDisabled)
    return kLogDisabled;
  if (value > kLogDebug)
    return kLogDebug;
  return value;
}

void signal_backtrace_handler(int signo) {
  if (!g_signal_backtrace_enabled.exchange(false, std::memory_order_relaxed))
    _exit(128 + signo);

  const char header[] = "\n[rocjitsu-hooks] signal backtrace\n";
  (void)::write(STDERR_FILENO, header, sizeof(header) - 1);
  void *frames[128];
  int count = ::backtrace(frames, 128);
  ::backtrace_symbols_fd(frames, count, STDERR_FILENO);
  ::signal(signo, SIG_DFL);
  ::raise(signo);
}

void maybe_install_signal_backtrace(bool enabled) {
  if (!enabled)
    return;

  g_signal_backtrace_enabled.store(true, std::memory_order_relaxed);
  ::signal(SIGSEGV, signal_backtrace_handler);
  ::signal(SIGABRT, signal_backtrace_handler);
}

[[nodiscard]] std::optional<HookConfig> parse_config() {
  std::optional<rocjitsu::config::DbtGuestConfig> dbt_guest;
  try {
    dbt_guest = rocjitsu::config::load_dbt_guest_config_from_rj_config();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to load RJ_CONFIG: %s\n", error.what());
    return std::nullopt;
  }
  if (!dbt_guest) {
    std::fprintf(stderr, "[rocjitsu-hooks] RJ_CONFIG is required for the DBT HSA hook\n");
    return std::nullopt;
  }

  if (!dbt_guest->enabled) {
    std::fprintf(stderr, "[rocjitsu-hooks] RJ_CONFIG does not enable dbt_guest mode\n");
    return std::nullopt;
  }

  auto target = parse_target(dbt_guest->host_isa);
  if (!target) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid dbt_guest.host_isa='%s'\n",
                 dbt_guest->host_isa.c_str());
    return std::nullopt;
  }
  auto guest = parse_target(dbt_guest->guest_isa);
  if (!guest) {
    std::fprintf(stderr, "[rocjitsu-hooks] invalid dbt_guest.guest_isa='%s'\n",
                 dbt_guest->guest_isa.c_str());
    return std::nullopt;
  }

  HookConfig config;
  config.target = *target;
  config.source_override = *guest;
  config.guest_target = *guest;
  config.host_gpu_id = dbt_guest->host_gpu_id;
  config.log_level = clamp_log_level(dbt_guest->log_level);
  config.signal_backtrace = dbt_guest->signal_backtrace;
  return config;
}

void log_message(int required_level, const char *format, ...) {
  if (g_log_level.load(std::memory_order_relaxed) < required_level)
    return;

  std::array<char, 512> message{};
  va_list args;
  va_start(args, format);
  std::vsnprintf(message.data(), message.size(), format, args);
  va_end(args);

  util::Logger::dbt_hooks(message.data());
}

void trace_message(int required_level, const char *format, ...) {
  if (g_log_level.load(std::memory_order_relaxed) < required_level)
    return;

  std::fprintf(stderr, "[rocjitsu-hooks] ");
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
}

[[nodiscard]] const char *arch_name(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return "cdna3";
  case ROCJITSU_CODE_ARCH_CDNA4:
    return "cdna4";
  case ROCJITSU_CODE_ARCH_RDNA3:
    return "rdna3";
  case ROCJITSU_CODE_ARCH_RDNA4:
    return "rdna4";
  default:
    return "invalid";
  }
}

[[nodiscard]] const char *diagnostic_severity_name(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  }
  return "diagnostic";
}

[[nodiscard]] const char *diagnostic_kind_name(DiagnosticKind kind) {
  switch (kind) {
  case DiagnosticKind::UnsupportedGuestArch:
    return "unsupported-guest-arch";
  case DiagnosticKind::KernelDescriptor:
    return "kernel-descriptor";
  case DiagnosticKind::Legalization:
    return "legalization";
  case DiagnosticKind::ExpandMissing:
    return "expand-missing";
  case DiagnosticKind::ExpandFailed:
    return "expand-failed";
  case DiagnosticKind::ResourceLimit:
    return "resource-limit";
  }
  return "unknown";
}

/// @brief Print one structured DBT diagnostic in the same compact style as the CLI.
void print_diagnostic(FILE *stream, const TranslationDiagnostic &diagnostic) {
  std::fprintf(stream, "[rocjitsu-dbt] %s: %s", diagnostic_severity_name(diagnostic.severity),
               diagnostic_kind_name(diagnostic.kind));
  if (diagnostic.guest_offset)
    std::fprintf(stream, " .text+0x%llx",
                 static_cast<unsigned long long>(*diagnostic.guest_offset));
  if (!diagnostic.mnemonic.empty())
    std::fprintf(stream, " %s", diagnostic.mnemonic.c_str());
  std::fprintf(stream, ": %s\n", diagnostic.message.c_str());
  for (const std::string &item : diagnostic.required_work)
    std::fprintf(stream, "[rocjitsu-dbt]   required: %s\n", item.c_str());
}

/// @brief Print errors unconditionally and lower-severity diagnostics when requested.
void print_diagnostics(FILE *stream, std::span<const TranslationDiagnostic> diagnostics,
                       bool include_warnings) {
  for (const TranslationDiagnostic &diagnostic : diagnostics) {
    if (diagnostic.severity == DiagnosticSeverity::Error || include_warnings)
      print_diagnostic(stream, diagnostic);
  }
}

struct DetectedElfTarget {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t mach = 0;
};

/// @brief Detect the rocjitsu ISA family and exact ELF MACH from an AMDGPU ELF header.
///
/// @details HSA code-object readers are opaque once created, so source ISA
/// detection has to use the ELF bytes captured at reader creation time. The
/// helper checks only ELF identity, machine type, and `EF_AMDGPU_MACH`. The
/// exact MACH is kept because ROCR rejects same-family-but-different-stepping
/// code objects such as gfx1200 ELFs loaded on gfx1201 agents.
[[nodiscard]] DetectedElfTarget detect_target_from_elf(const uint8_t *bytes, size_t size) {
  if (bytes == nullptr || size < sizeof(Elf64_Ehdr))
    return {};

  Elf64_Ehdr header{};
  std::memcpy(&header, bytes, sizeof(header));
  if (std::memcmp(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE) != 0)
    return {};
  if (header.e_ident[rocjitsu::EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU)
    return {};

  const uint32_t mach = header.e_flags & EF_AMDGPU_MACH;
  return DetectedElfTarget{arch_for_elf_mach(mach), mach};
}

/// @brief Process-local map from HSA code-object reader handles to ELF bytes.
///
/// @details `hsa_executable_load_agent_code_object()` receives only an opaque
/// reader handle. The create wrapper records memory-backed reader bytes here so
/// the load wrapper can translate the original ELF. The registry uses rocjitsu's
/// intrusive list and fixed-block arena so this C ABI path can report registry
/// exhaustion as an HSA status instead of depending on throwing STL allocation.
/// Entries for application readers are non-owning and rely on the application's
/// reader lifetime. Entries for hidden translated readers own a vector so ROCR's
/// memory-reader pointer remains valid while the translated load is in progress.
class CodeObjectReaderRegistry {
public:
  /// @brief Return the singleton registry used by all hook wrappers.
  static CodeObjectReaderRegistry &instance() {
    static CodeObjectReaderRegistry registry;
    return registry;
  }

  /// @brief Record bytes backing a code-object reader.
  /// @param reader HSA reader handle used as the lookup key.
  /// @param bytes Start of the ELF image.
  /// @param size Size of the ELF image in bytes.
  /// @param owned Optional owned storage for translated ELF bytes.
  [[nodiscard]] bool store(hsa_code_object_reader_t reader, const uint8_t *bytes, size_t size,
                           std::vector<uint8_t> *owned) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        delete entry->owned;
        entry->bytes = bytes;
        entry->size = size;
        entry->owned = owned;
        return true;
      }
    }

    void *storage = entry_pool_.try_allocate(sizeof(Entry));
    if (storage == nullptr)
      return false;
    auto *entry = new (storage) Entry(reader.handle, bytes, size, owned);
    entries_.push_front(*entry);
    return true;
  }

  /// @brief Find bytes previously recorded for @p reader.
  /// @returns true when @p bytes and @p size were populated.
  bool lookup(hsa_code_object_reader_t reader, const uint8_t **bytes, size_t *size) {
    std::shared_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        *bytes = entry->bytes;
        *size = entry->size;
        return true;
      }
    }
    return false;
  }

  /// @brief Remove one reader entry and release owned translated bytes if any.
  void remove(hsa_code_object_reader_t reader) {
    std::unique_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      auto *entry = static_cast<Entry *>(it.node_pointer());
      if (entry->handle == reader.handle) {
        it = entries_.erase(it);
        destroy_entry(entry);
        return;
      }
      ++it;
    }
  }

  /// @brief Clear all reader entries during tool unload.
  void clear() {
    std::unique_lock lock(mutex_);
    while (!entries_.empty()) {
      auto it = entries_.begin();
      auto *entry = static_cast<Entry *>(it.node_pointer());
      entries_.erase(it);
      destroy_entry(entry);
    }
  }

private:
  struct Entry : util::IListNode<Entry> {
    Entry(uint64_t h, const uint8_t *b, size_t s, std::vector<uint8_t> *o)
        : handle(h), bytes(b), size(s), owned(o) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::vector<uint8_t> *owned = nullptr;
  };

  void destroy_entry(Entry *entry) {
    delete entry->owned;
    entry->~Entry();
    entry_pool_.deallocate(entry);
  }

  mutable std::shared_mutex mutex_;
  util::ArenaAlloc<sizeof(Entry), 256, alignof(Entry)> entry_pool_;
  util::IntrusiveList<Entry> entries_;
};

class ExecutableAgentRegistry {
public:
  static ExecutableAgentRegistry &instance() {
    static ExecutableAgentRegistry registry;
    return registry;
  }

  void record(hsa_executable_t executable, hsa_agent_t guest, hsa_agent_t host) {
    std::lock_guard lock(mutex_);
    map_[key(executable, guest)] = host.handle;
  }

  hsa_agent_t map_agent(hsa_executable_t executable, hsa_agent_t agent) {
    std::lock_guard lock(mutex_);
    auto it = map_.find(key(executable, agent));
    if (it == map_.end())
      return agent;
    return hsa_agent_t{it->second};
  }

  void erase_executable(hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    const uint64_t prefix = executable.handle;
    for (auto it = map_.begin(); it != map_.end();) {
      if (it->first.first == prefix)
        it = map_.erase(it);
      else
        ++it;
    }
  }

  void clear() {
    std::lock_guard lock(mutex_);
    map_.clear();
  }

private:
  using Key = std::pair<uint64_t, uint64_t>;

  struct KeyHash {
    size_t operator()(Key key) const {
      size_t h = std::hash<uint64_t>{}(key.first);
      h ^= std::hash<uint64_t>{}(key.second) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  static Key key(hsa_executable_t executable, hsa_agent_t agent) {
    return {executable.handle, agent.handle};
  }

  std::mutex mutex_;
  std::unordered_map<Key, uint64_t, KeyHash> map_;
};

/// @brief Record memory-backed HSA code-object bytes for later DBT translation.
hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader);

/// @brief Warn for file-backed readers, which the MVP cannot translate safely.
hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader);

/// @brief Remove tracked reader bytes and forward reader destruction.
hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader);

/// @brief Skip ROCR shutdown in guest mode and uninstall rocjitsu wrappers.
hsa_status_t HSA_API rj_shut_down();

/// @brief Shadow public HSA agent iteration with the guest replacing the host.
hsa_status_t HSA_API rj_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void *data),
                                       void *data);

/// @brief Translate guest code objects and load the translated ELF on the host agent.
hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object);

/// @brief Preserve guest ISA iteration so framework fatbin selection picks guest images.
hsa_status_t HSA_API rj_agent_iterate_isas(hsa_agent_t agent,
                                           hsa_status_t (*callback)(hsa_isa_t isa, void *data),
                                           void *data);

/// @brief Create real host queues when the application asks for a guest queue.
hsa_status_t HSA_API rj_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                     void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                     void *data, uint32_t private_segment_size,
                                     uint32_t group_segment_size, hsa_queue_t **queue);

/// @brief Forward destruction for queues returned by guest queue creation.
hsa_status_t HSA_API rj_queue_destroy(hsa_queue_t *queue);

/// @brief Return host memory regions for guest region iteration.
hsa_status_t HSA_API rj_agent_iterate_regions(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void *data), void *data);

/// @brief Assign memory access to the host agent when the caller passes the guest.
hsa_status_t HSA_API rj_memory_assign_agent(void *ptr, hsa_agent_t agent,
                                            hsa_access_permission_t access);

/// @brief Drop executable-agent mappings before forwarding executable destruction.
hsa_status_t HSA_API rj_executable_destroy(hsa_executable_t executable);

/// @brief Remap guest symbol queries to the host load agent.
hsa_status_t HSA_API rj_executable_get_symbol(hsa_executable_t executable, const char *module_name,
                                              const char *symbol_name, hsa_agent_t agent,
                                              int32_t call_convention,
                                              hsa_executable_symbol_t *symbol);

/// @brief Remap by-name guest symbol queries to the host load agent.
hsa_status_t HSA_API rj_executable_get_symbol_by_name(hsa_executable_t executable,
                                                      const char *symbol_name,
                                                      const hsa_agent_t *agent,
                                                      hsa_executable_symbol_t *symbol);

/// @brief Define executable globals against the host load agent.
hsa_status_t HSA_API rj_executable_agent_global_variable_define(hsa_executable_t executable,
                                                                hsa_agent_t agent,
                                                                const char *variable_name,
                                                                void *address);

/// @brief Iterate host-agent symbols while reporting the guest agent to callbacks.
hsa_status_t HSA_API rj_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data);

/// @brief Enumerate guest memory pools for public guest-agent discovery.
hsa_status_t HSA_API rj_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data);

/// @brief Return memory-pool properties without changing guest-facing identity.
hsa_status_t HSA_API rj_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                                 hsa_amd_memory_pool_info_t attribute, void *value);

/// @brief Allocate from a matching host pool when the caller passes a guest pool.
hsa_status_t HSA_API rj_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                 uint32_t flags, void **ptr);

/// @brief Free host-backed memory allocated through mapped pools.
hsa_status_t HSA_API rj_amd_memory_pool_free(void *ptr);

/// @brief Query agent/pool relationships through mapped host handles.
hsa_status_t HSA_API rj_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                       hsa_amd_memory_pool_t memory_pool,
                                                       hsa_amd_agent_memory_pool_info_t attribute,
                                                       void *value);

/// @brief Allow memory access for mapped host agents, removing duplicates.
hsa_status_t HSA_API rj_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t *agents,
                                                const uint32_t *flags, const void *ptr);

/// @brief Map source and destination agents for async memory copies.
hsa_status_t HSA_API rj_amd_memory_async_copy(void *dst, hsa_agent_t dst_agent, const void *src,
                                              hsa_agent_t src_agent, size_t size,
                                              uint32_t num_dep_signals,
                                              const hsa_signal_t *dep_signals,
                                              hsa_signal_t completion_signal);

/// @brief Map source and destination agents for engine-selected async copies.
hsa_status_t HSA_API rj_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t *dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma);

/// @brief Map the copy agent for rectangular async copies.
hsa_status_t HSA_API rj_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t *dst, const hsa_dim3_t *dst_offset, const hsa_pitched_ptr_t *src,
    const hsa_dim3_t *src_offset, const hsa_dim3_t *range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal);

/// @brief Query copy-engine status using mapped host agents.
hsa_status_t HSA_API rj_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                                      uint32_t *engine_ids_mask);

/// @brief Query preferred copy engines using mapped host agents.
hsa_status_t HSA_API rj_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
                                                             hsa_agent_t src_agent,
                                                             hsa_amd_sdma_engine_id_t *engine_id);

/// @brief Map agent arrays before locking host memory for GPU access.
hsa_status_t HSA_API rj_amd_memory_lock(void *host_ptr, size_t size, hsa_agent_t *agents,
                                        int num_agent, void **agent_ptr);

/// @brief Forward memory-fill operations while preserving trace visibility.
hsa_status_t HSA_API rj_amd_memory_fill(void *ptr, uint32_t value, size_t count);

/// @brief Map pointer-info owner and accessible agents from host back to guest.
hsa_status_t HSA_API rj_amd_pointer_info(const void *ptr, hsa_amd_pointer_info_t *info,
                                         void *(*alloc)(size_t), uint32_t *num_agents_accessible,
                                         hsa_agent_t **accessible);

/// @brief Map agent arrays and guest pools before locking host memory to a pool.
hsa_status_t HSA_API rj_amd_memory_lock_to_pool(void *host_ptr, size_t size, hsa_agent_t *agents,
                                                int num_agent, hsa_amd_memory_pool_t pool,
                                                uint32_t flags, void **agent_ptr);

/// @brief Prefetch SVM memory to the host agent when the caller passes the guest.
hsa_status_t HSA_API rj_amd_svm_prefetch_async(void *ptr, size_t size, hsa_agent_t agent,
                                               uint32_t num_dep_signals,
                                               const hsa_signal_t *dep_signals,
                                               hsa_signal_t completion_signal);

/// @brief Rewrite virtual-memory access descriptors from guest to host.
hsa_status_t HSA_API rj_amd_vmem_set_access(void *va, size_t size,
                                            const hsa_amd_memory_access_desc_t *desc,
                                            size_t desc_cnt);

/// @brief Query virtual-memory access through the host agent.
hsa_status_t HSA_API rj_amd_vmem_get_access(void *va, hsa_access_permission_t *perms,
                                            hsa_agent_t agent_handle);

/// @brief Apply async scratch limits to the real host agent.
hsa_status_t HSA_API rj_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold);

/// @brief Map every embedded agent in AMD batch-copy operations.
hsa_status_t HSA_API rj_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t *copy_ops,
                                                    uint32_t num_copy_ops, uint32_t num_dep_signals,
                                                    const hsa_signal_t *dep_signals);

/// @brief Preload runtime state on the real host agent.
hsa_status_t HSA_API rj_amd_agent_preload(hsa_agent_t agent, uint64_t flags);

/// @brief Clear cached guest-to-host memory-pool mappings on HSA unload.
void clear_memory_pool_mapper();

/// @brief Original HSA table entries saved for internal hook queries but not patched.
///
/// @details Each entry is:
/// `X(name, table_ptr, present, field, type)`.
/// - `name` is the `RjHsaLayer` getter name and `original_<name>_` member stem.
/// - `table_ptr` is the API-table member pointer that owns `field`.
/// - `present` guards table access when a table or versioned field may be absent.
/// - `field` is the HSA table function-pointer field to save.
/// - `type` is the exact function-pointer type stored in `original_<name>_`.
#define RJ_HSA_SAVED_ONLY_ENTRIES(X)                                                               \
  X(agent_get_info, core_, true, hsa_agent_get_info_fn, decltype(hsa_agent_get_info) *)            \
  X(isa_get_info_alt, core_, true, hsa_isa_get_info_alt_fn, decltype(hsa_isa_get_info_alt) *)

/// @brief HSA table entries patched by the DBT hook.
///
/// @details Each entry is:
/// `X(name, table_ptr, present, patch_if_original, field, wrapper, type)`.
/// - `name` is the `RjHsaLayer` getter name and `original_<name>_` member stem.
/// - `table_ptr` is the API-table member pointer that owns `field`.
/// - `present` guards table access when a table or versioned field may be absent.
/// - `patch_if_original` means install only writes `wrapper` when the saved
///   original function pointer is non-null. Core entries set this false because
///   they are required after `validate_table()`. AMD extension entries set this
///   true because ROCR may expose the table field but leave the function absent.
/// - `field` is the HSA table function-pointer field to save/patch/restore.
/// - `wrapper` is the rocjitsu replacement function assigned during install.
/// - `type` is the exact function-pointer type stored in `original_<name>_`.
#define RJ_HSA_PATCH_ENTRIES(X)                                                                    \
  X(shut_down, core_, true, false, hsa_shut_down_fn, rj_shut_down, decltype(hsa_shut_down) *)      \
  X(iterate_agents, core_, true, false, hsa_iterate_agents_fn, rj_iterate_agents,                  \
    decltype(hsa_iterate_agents) *)                                                                \
  X(agent_iterate_isas, core_, true, false, hsa_agent_iterate_isas_fn, rj_agent_iterate_isas,      \
    decltype(hsa_agent_iterate_isas) *)                                                            \
  X(queue_create, core_, true, false, hsa_queue_create_fn, rj_queue_create,                        \
    decltype(hsa_queue_create) *)                                                                  \
  X(queue_destroy, core_, true, false, hsa_queue_destroy_fn, rj_queue_destroy,                     \
    decltype(hsa_queue_destroy) *)                                                                 \
  X(agent_iterate_regions, core_, true, false, hsa_agent_iterate_regions_fn,                       \
    rj_agent_iterate_regions, decltype(hsa_agent_iterate_regions) *)                               \
  X(memory_assign_agent, core_, true, false, hsa_memory_assign_agent_fn, rj_memory_assign_agent,   \
    decltype(hsa_memory_assign_agent) *)                                                           \
  X(executable_destroy, core_, true, false, hsa_executable_destroy_fn, rj_executable_destroy,      \
    decltype(hsa_executable_destroy) *)                                                            \
  X(executable_get_symbol, core_, true, false, hsa_executable_get_symbol_fn,                       \
    rj_executable_get_symbol, decltype(hsa_executable_get_symbol) *)                               \
  X(executable_get_symbol_by_name, core_, true, false, hsa_executable_get_symbol_by_name_fn,       \
    rj_executable_get_symbol_by_name, decltype(hsa_executable_get_symbol_by_name) *)               \
  X(executable_agent_global_variable_define, core_, true, false,                                   \
    hsa_executable_agent_global_variable_define_fn, rj_executable_agent_global_variable_define,    \
    decltype(hsa_executable_agent_global_variable_define) *)                                       \
  X(executable_iterate_agent_symbols, core_, true, false, hsa_executable_iterate_agent_symbols_fn, \
    rj_executable_iterate_agent_symbols, decltype(hsa_executable_iterate_agent_symbols) *)         \
  X(create_from_file, core_, true, false, hsa_code_object_reader_create_from_file_fn,              \
    rj_code_object_reader_create_from_file, decltype(hsa_code_object_reader_create_from_file) *)   \
  X(create_from_memory, core_, true, false, hsa_code_object_reader_create_from_memory_fn,          \
    rj_code_object_reader_create_from_memory,                                                      \
    decltype(hsa_code_object_reader_create_from_memory) *)                                         \
  X(destroy, core_, true, false, hsa_code_object_reader_destroy_fn, rj_code_object_reader_destroy, \
    decltype(hsa_code_object_reader_destroy) *)                                                    \
  X(load_agent_code_object, core_, true, false, hsa_executable_load_agent_code_object_fn,          \
    rj_executable_load_agent_code_object, decltype(hsa_executable_load_agent_code_object) *)       \
  X(amd_memory_pool_get_info, amd_ext_, amd_ext_ != nullptr, true,                                 \
    hsa_amd_memory_pool_get_info_fn, rj_amd_memory_pool_get_info,                                  \
    hsa_amd_memory_pool_get_info_fn_t)                                                             \
  X(amd_agent_iterate_memory_pools, amd_ext_, amd_ext_ != nullptr, true,                           \
    hsa_amd_agent_iterate_memory_pools_fn, rj_amd_agent_iterate_memory_pools,                      \
    hsa_amd_agent_iterate_memory_pools_fn_t)                                                       \
  X(amd_memory_pool_allocate, amd_ext_, amd_ext_ != nullptr, true,                                 \
    hsa_amd_memory_pool_allocate_fn, rj_amd_memory_pool_allocate,                                  \
    hsa_amd_memory_pool_allocate_fn_t)                                                             \
  X(amd_memory_pool_free, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_pool_free_fn,        \
    rj_amd_memory_pool_free, hsa_amd_memory_pool_free_fn_t)                                        \
  X(amd_agent_memory_pool_get_info, amd_ext_, amd_ext_ != nullptr, true,                           \
    hsa_amd_agent_memory_pool_get_info_fn, rj_amd_agent_memory_pool_get_info,                      \
    hsa_amd_agent_memory_pool_get_info_fn_t)                                                       \
  X(amd_agents_allow_access, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_agents_allow_access_fn,  \
    rj_amd_agents_allow_access, hsa_amd_agents_allow_access_fn_t)                                  \
  X(amd_memory_async_copy, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_async_copy_fn,      \
    rj_amd_memory_async_copy, hsa_amd_memory_async_copy_fn_t)                                      \
  X(amd_memory_async_copy_on_engine, amd_ext_, amd_ext_ != nullptr, true,                          \
    hsa_amd_memory_async_copy_on_engine_fn, rj_amd_memory_async_copy_on_engine,                    \
    hsa_amd_memory_async_copy_on_engine_fn_t)                                                      \
  X(amd_memory_async_copy_rect, amd_ext_, amd_ext_ != nullptr, true,                               \
    hsa_amd_memory_async_copy_rect_fn, rj_amd_memory_async_copy_rect,                              \
    hsa_amd_memory_async_copy_rect_fn_t)                                                           \
  X(amd_memory_copy_engine_status, amd_ext_, amd_ext_ != nullptr, true,                            \
    hsa_amd_memory_copy_engine_status_fn, rj_amd_memory_copy_engine_status,                        \
    hsa_amd_memory_copy_engine_status_fn_t)                                                        \
  X(amd_memory_lock, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_lock_fn,                  \
    rj_amd_memory_lock, hsa_amd_memory_lock_fn_t)                                                  \
  X(amd_memory_lock_to_pool, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_lock_to_pool_fn,  \
    rj_amd_memory_lock_to_pool, hsa_amd_memory_lock_to_pool_fn_t)                                  \
  X(amd_svm_prefetch_async, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_svm_prefetch_async_fn,    \
    rj_amd_svm_prefetch_async, hsa_amd_svm_prefetch_async_fn_t)                                    \
  X(amd_pointer_info, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_pointer_info_fn,                \
    rj_amd_pointer_info, hsa_amd_pointer_info_fn_t)                                                \
  X(amd_vmem_set_access, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_vmem_set_access_fn,          \
    rj_amd_vmem_set_access, hsa_amd_vmem_set_access_fn_t)                                          \
  X(amd_vmem_get_access, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_vmem_get_access_fn,          \
    rj_amd_vmem_get_access, hsa_amd_vmem_get_access_fn_t)                                          \
  X(amd_agent_set_async_scratch_limit, amd_ext_, amd_ext_ != nullptr, true,                        \
    hsa_amd_agent_set_async_scratch_limit_fn, rj_amd_agent_set_async_scratch_limit,                \
    hsa_amd_agent_set_async_scratch_limit_fn_t)                                                    \
  X(amd_memory_get_preferred_copy_engine, amd_ext_, amd_ext_ != nullptr, true,                     \
    hsa_amd_memory_get_preferred_copy_engine_fn, rj_amd_memory_get_preferred_copy_engine,          \
    hsa_amd_memory_get_preferred_copy_engine_fn_t)                                                 \
  X(amd_memory_fill, amd_ext_, amd_ext_ != nullptr, true, hsa_amd_memory_fill_fn,                  \
    rj_amd_memory_fill, hsa_amd_memory_fill_fn_t)                                                  \
  X(amd_memory_async_batch_copy, amd_ext_,                                                         \
    amd_ext_ != nullptr && RJ_AMD_EXT_HAS_FIELD(amd_ext_, hsa_amd_memory_async_batch_copy_fn),     \
    true, hsa_amd_memory_async_batch_copy_fn, rj_amd_memory_async_batch_copy,                      \
    hsa_amd_memory_async_batch_copy_fn_t)                                                          \
  X(amd_agent_preload, amd_ext_,                                                                   \
    amd_ext_ != nullptr && RJ_AMD_EXT_HAS_FIELD(amd_ext_, hsa_amd_agent_preload_fn), true,         \
    hsa_amd_agent_preload_fn, rj_amd_agent_preload, hsa_amd_agent_preload_fn_t)

/// @brief Process-local HSA API table patch state for the rocjitsu DBT tool.
///
/// @details Tool chaining depends on saving the function pointers that are
/// present at `OnLoad()` time and calling those saved pointers from wrappers.
/// They may already point at another tool's wrapper. `OnUnload()` restores only
/// entries that still point at rocjitsu wrappers so later tools are not
/// accidentally overwritten.
class RjHsaLayer {
public:
  /// @brief Validate the incoming table, save original entries, and install wrappers.
  bool install(HsaApiTable *table, HookConfig config) {
    std::lock_guard lock(mutex_);
    if (active_) {
      std::fprintf(stderr, "[rocjitsu-hooks] OnLoad called while hook is already active\n");
      return false;
    }
    if (!validate_table(table))
      return false;

    table_ = table;
    core_ = table->core_;
    amd_ext_ = table->amd_ext_;
    g_log_level.store(config.log_level, std::memory_order_relaxed);
    config_ = std::move(config);

#define RJ_SAVE_SAVED_ONLY(name, table_ptr, present, field, type)                                  \
  if (present)                                                                                     \
    original_##name##_ = (table_ptr)->field;
    RJ_HSA_SAVED_ONLY_ENTRIES(RJ_SAVE_SAVED_ONLY)
#undef RJ_SAVE_SAVED_ONLY

#define RJ_SAVE_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)           \
  if (present)                                                                                     \
    original_##name##_ = (table_ptr)->field;
    RJ_HSA_PATCH_ENTRIES(RJ_SAVE_PATCH)
#undef RJ_SAVE_PATCH

    if (original_create_from_file_ == nullptr || original_create_from_memory_ == nullptr ||
        original_destroy_ == nullptr || original_load_agent_code_object_ == nullptr ||
        original_iterate_agents_ == nullptr || original_agent_get_info_ == nullptr ||
        original_agent_iterate_isas_ == nullptr || original_isa_get_info_alt_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] HSA core table contains null code-object entries\n");
      clear_unlocked();
      return false;
    }

#define RJ_INSTALL_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)        \
  if ((present) && (!(patch_if_original) || original_##name##_ != nullptr))                        \
    (table_ptr)->field = wrapper;
    RJ_HSA_PATCH_ENTRIES(RJ_INSTALL_PATCH)
#undef RJ_INSTALL_PATCH

    active_ = true;

    log_message(kLogInfo, "installed DBT hook target=%s arch=%s mach=0x%x",
                config_->target.name.data(), arch_name(config_->target.arch), config_->target.mach);
    return true;
  }

  /// @brief Restore rocjitsu wrappers if still installed and clear owned state.
  void uninstall() {
    std::lock_guard lock(mutex_);
    if (active_ && core_ != nullptr) {
      trace_message(kLogVerbose, "uninstall begin");
#define RJ_RESTORE_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)        \
  if ((present) && (table_ptr)->field == wrapper)                                                  \
    (table_ptr)->field = original_##name##_;
      RJ_HSA_PATCH_ENTRIES(RJ_RESTORE_PATCH)
#undef RJ_RESTORE_PATCH
    }

    CodeObjectReaderRegistry::instance().clear();
    ExecutableAgentRegistry::instance().clear();
    clear_memory_pool_mapper();
    trace_message(kLogVerbose, "uninstall end");
    clear_unlocked();
  }

  [[nodiscard]] int log_level() const {
    std::lock_guard lock(mutex_);
    return config_ ? config_->log_level : kLogDisabled;
  }

  [[nodiscard]] std::optional<HookConfig> config() const {
    std::lock_guard lock(mutex_);
    return config_;
  }

#define RJ_DEFINE_SAVED_ONLY_GETTER(name, table_ptr, present, field, type)                         \
  [[nodiscard]] type name() const {                                                                \
    std::lock_guard lock(mutex_);                                                                  \
    return original_##name##_;                                                                     \
  }
  RJ_HSA_SAVED_ONLY_ENTRIES(RJ_DEFINE_SAVED_ONLY_GETTER)
#undef RJ_DEFINE_SAVED_ONLY_GETTER

#define RJ_DEFINE_PATCH_GETTER(name, table_ptr, present, patch_if_original, field, wrapper, type)  \
  [[nodiscard]] type name() const {                                                                \
    std::lock_guard lock(mutex_);                                                                  \
    return original_##name##_;                                                                     \
  }
  RJ_HSA_PATCH_ENTRIES(RJ_DEFINE_PATCH_GETTER)
#undef RJ_DEFINE_PATCH_GETTER

private:
  [[nodiscard]] static bool validate_table(HsaApiTable *table) {
    if (table == nullptr || table->core_ == nullptr) {
      std::fprintf(stderr, "[rocjitsu-hooks] invalid HSA API table passed to OnLoad\n");
      return false;
    }

    constexpr size_t required_size =
        offsetof(CoreApiTable, hsa_executable_iterate_agent_symbols_fn) +
        sizeof(CoreApiTable::hsa_executable_iterate_agent_symbols_fn);
    if (table->core_->version.minor_id < required_size) {
      std::fprintf(stderr,
                   "[rocjitsu-hooks] HSA core table too small: got %u bytes, need %zu bytes\n",
                   table->core_->version.minor_id, required_size);
      return false;
    }
    return true;
  }

  void clear_unlocked() {
    active_ = false;
    g_log_level.store(kLogDisabled, std::memory_order_relaxed);
    g_signal_backtrace_enabled.store(false, std::memory_order_relaxed);
    table_ = nullptr;
    core_ = nullptr;
    amd_ext_ = nullptr;
    config_.reset();

#define RJ_CLEAR_SAVED_ONLY(name, table_ptr, present, field, type) original_##name##_ = nullptr;
    RJ_HSA_SAVED_ONLY_ENTRIES(RJ_CLEAR_SAVED_ONLY)
#undef RJ_CLEAR_SAVED_ONLY

#define RJ_CLEAR_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)          \
  original_##name##_ = nullptr;
    RJ_HSA_PATCH_ENTRIES(RJ_CLEAR_PATCH)
#undef RJ_CLEAR_PATCH
  }

  mutable std::mutex mutex_;
  HsaApiTable *table_ = nullptr;
  CoreApiTable *core_ = nullptr;
  AmdExtTable *amd_ext_ = nullptr;
  std::optional<HookConfig> config_;
  bool active_ = false;

#define RJ_DECLARE_SAVED_ONLY(name, table_ptr, present, field, type)                               \
  type original_##name##_ = nullptr;
  RJ_HSA_SAVED_ONLY_ENTRIES(RJ_DECLARE_SAVED_ONLY)
#undef RJ_DECLARE_SAVED_ONLY

#define RJ_DECLARE_PATCH(name, table_ptr, present, patch_if_original, field, wrapper, type)        \
  type original_##name##_ = nullptr;
  RJ_HSA_PATCH_ENTRIES(RJ_DECLARE_PATCH)
#undef RJ_DECLARE_PATCH
};

RjHsaLayer &layer() {
  static RjHsaLayer state;
  return state;
}

[[nodiscard]] bool parse_u32_text(const char *text, uint32_t *out) {
  if (!text || !out)
    return false;
  errno = 0;
  char *end = nullptr;
  unsigned long parsed = std::strtoul(text, &end, 0);
  if (errno != 0 || end == text || parsed > UINT32_MAX)
    return false;
  while (*end == '\n' || *end == '\r' || *end == ' ' || *end == '\t')
    ++end;
  if (*end != '\0')
    return false;
  *out = static_cast<uint32_t>(parsed);
  return true;
}

[[nodiscard]] std::optional<uint32_t> read_u32_file(const std::string &path) {
  FILE *file = std::fopen(path.c_str(), "r");
  if (!file)
    return std::nullopt;

  std::array<char, 64> buffer{};
  char *line = std::fgets(buffer.data(), static_cast<int>(buffer.size()), file);
  std::fclose(file);
  if (!line)
    return std::nullopt;

  uint32_t value = 0;
  if (!parse_u32_text(buffer.data(), &value))
    return std::nullopt;
  return value;
}

[[nodiscard]] std::optional<uint32_t> node_id_for_kfd_gpu_id_in_root(const char *root,
                                                                     uint32_t gpu_id) {
  DIR *dir = opendir(root);
  if (!dir)
    return std::nullopt;

  std::optional<uint32_t> result;
  while (dirent *entry = readdir(dir)) {
    if (entry->d_name[0] == '.')
      continue;

    uint32_t node_id = 0;
    if (!parse_u32_text(entry->d_name, &node_id))
      continue;

    std::string gpu_id_path = std::string(root) + "/" + entry->d_name + "/gpu_id";
    std::optional<uint32_t> node_gpu_id = read_u32_file(gpu_id_path);
    if (node_gpu_id && *node_gpu_id == gpu_id) {
      result = node_id;
      break;
    }
  }
  closedir(dir);
  return result;
}

/// @brief Translate a configured KFD gpu_id to ROCR's HSA driver node id.
///
/// @details The config uses the KFD topology `gpu_id` because it is stable
/// across ROCR agent handle creation and is the id KFD ioctls use. HSA does
/// not expose that value directly, so the hook reads the redirected topology
/// tree and later compares agents by `HSA_AMD_AGENT_INFO_DRIVER_NODE_ID`.
[[nodiscard]] std::optional<uint32_t> node_id_for_kfd_gpu_id(uint32_t gpu_id) {
  constexpr std::array<const char *, 2> kTopologyNodeRoots = {
      "/sys/devices/virtual/kfd/kfd/topology/nodes", "/sys/class/kfd/kfd/topology/nodes"};
  for (const char *root : kTopologyNodeRoots) {
    if (std::optional<uint32_t> node_id = node_id_for_kfd_gpu_id_in_root(root, gpu_id))
      return node_id;
  }
  return std::nullopt;
}

class AgentMapper {
public:
  static AgentMapper &instance() {
    static AgentMapper mapper;
    return mapper;
  }

  hsa_agent_t map(hsa_agent_t agent) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    if (has_mapping_ && agent.handle == guest_.handle)
      return host_;
    return agent;
  }

  bool is_guest(hsa_agent_t agent) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    return has_mapping_ && agent.handle == guest_.handle;
  }

  hsa_agent_t host_for_guest() {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    return has_mapping_ ? host_ : hsa_agent_t{};
  }

  hsa_agent_t guest_agent() {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    return has_mapping_ ? guest_ : hsa_agent_t{};
  }

  hsa_agent_t guest_for_host(hsa_agent_t agent) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    if (has_mapping_ && agent.handle == host_.handle)
      return guest_;
    return agent;
  }

private:
  struct AgentSearchData {
    AgentMapper *self = nullptr;
    TargetInfo guest{};
    TargetInfo host{};
    std::optional<uint32_t> host_node_id;
  };

  struct IsaSearchData {
    AgentMapper *self = nullptr;
    TargetInfo target{};
    bool found = false;
  };

  static bool target_matches_isa_name(std::string_view isa_name, std::string_view target) {
    constexpr std::string_view prefix = "amdgcn-amd-amdhsa--";
    if (isa_name.starts_with(prefix))
      isa_name.remove_prefix(prefix.size());
    if (!isa_name.starts_with(target))
      return false;
    if (isa_name.size() == target.size())
      return true;
    return isa_name[target.size()] == ':' || isa_name[target.size()] == '\0';
  }

  static hsa_status_t isa_callback(hsa_isa_t isa, void *data) {
    auto *search = static_cast<IsaSearchData *>(data);
    auto *get_info = layer().isa_get_info_alt();
    if (!get_info)
      return HSA_STATUS_ERROR;

    uint32_t name_length = 0;
    if (get_info(isa, HSA_ISA_INFO_NAME_LENGTH, &name_length) != HSA_STATUS_SUCCESS ||
        name_length == 0)
      return HSA_STATUS_SUCCESS;
    std::vector<char> name(name_length + 1, 0);
    if (get_info(isa, HSA_ISA_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    if (target_matches_isa_name(std::string_view(name.data()), search->target.name)) {
      search->found = true;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  }

  bool agent_has_target(hsa_agent_t agent, TargetInfo target) {
    auto *iterate_isas = layer().agent_iterate_isas();
    if (!iterate_isas)
      return false;
    IsaSearchData search{this, target, false};
    hsa_status_t status = iterate_isas(agent, isa_callback, &search);
    return search.found || status == HSA_STATUS_INFO_BREAK;
  }

  std::optional<uint32_t> agent_driver_node_id(hsa_agent_t agent) {
    auto *get_info = layer().agent_get_info();
    if (!get_info)
      return std::nullopt;

    uint32_t node_id = 0;
    hsa_status_t status =
        get_info(agent, static_cast<hsa_agent_info_t>(kHsaAmdAgentInfoDriverNodeId), &node_id);
    if (status != HSA_STATUS_SUCCESS)
      return std::nullopt;
    return node_id;
  }

  bool agent_has_driver_node(hsa_agent_t agent, uint32_t expected_node_id) {
    std::optional<uint32_t> node_id = agent_driver_node_id(agent);
    return node_id && *node_id == expected_node_id;
  }

  bool agent_matches_host(hsa_agent_t agent, const AgentSearchData &search) {
    if (!agent_has_target(agent, search.host))
      return false;
    if (search.host_node_id)
      return agent_has_driver_node(agent, *search.host_node_id);
    return true;
  }

  static hsa_status_t agent_callback(hsa_agent_t agent, void *data) {
    auto *search = static_cast<AgentSearchData *>(data);
    if (search->self->agent_has_target(agent, search->guest))
      search->self->guest_ = agent;
    if (search->self->host_.handle == 0 && search->self->agent_matches_host(agent, *search))
      search->self->host_ = agent;
    if (search->self->guest_.handle != 0 && search->self->host_.handle != 0)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  }

  void ensure_discovered() {
    std::lock_guard lock(mutex_);
    if (discovered_)
      return;
    discovered_ = true;

    auto config = layer().config();
    if (!config || !config->guest_target)
      return;

    auto *iterate_agents = layer().iterate_agents();
    if (!iterate_agents)
      return;

    std::optional<uint32_t> host_node_id;
    if (config->host_gpu_id != 0) {
      host_node_id = node_id_for_kfd_gpu_id(config->host_gpu_id);
      if (!host_node_id) {
        std::fprintf(stderr,
                     "[rocjitsu-hooks] failed to find topology node for host KFD gpu_id=%u\n",
                     config->host_gpu_id);
        return;
      }
    }

    AgentSearchData search{this, *config->guest_target, config->target, host_node_id};
    hsa_status_t status = iterate_agents(agent_callback, &search);
    has_mapping_ = (status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK) &&
                   guest_.handle != 0 && host_.handle != 0 && guest_.handle != host_.handle;
    if (has_mapping_) {
      log_message(kLogInfo, "mapped guest agent=%llu to host agent=%llu",
                  static_cast<unsigned long long>(guest_.handle),
                  static_cast<unsigned long long>(host_.handle));
      if (config->log_level > kLogDisabled) {
        std::optional<uint32_t> selected_node_id = agent_driver_node_id(host_);
        std::fprintf(stderr,
                     "[rocjitsu-hooks] selected host agent=%llu for guest agent=%llu "
                     "host_gpu_id=%u host_node_id=%u\n",
                     static_cast<unsigned long long>(host_.handle),
                     static_cast<unsigned long long>(guest_.handle), config->host_gpu_id,
                     selected_node_id.value_or(0));
      }
    } else {
      std::fprintf(stderr, "[rocjitsu-hooks] failed to find guest/host HSA agents for DBT\n");
    }
  }

  std::mutex mutex_;
  bool discovered_ = false;
  bool has_mapping_ = false;
  hsa_agent_t guest_{};
  hsa_agent_t host_{};
};

class MemoryPoolMapper {
public:
  static MemoryPoolMapper &instance() {
    static MemoryPoolMapper mapper;
    return mapper;
  }

  hsa_amd_memory_pool_t map(hsa_amd_memory_pool_t pool) {
    ensure_discovered();
    std::lock_guard lock(mutex_);
    auto it = guest_to_host_.find(pool.handle);
    if (it == guest_to_host_.end())
      return pool;
    return hsa_amd_memory_pool_t{it->second};
  }

  void clear() {
    std::lock_guard lock(mutex_);
    discovered_ = false;
    guest_to_host_.clear();
  }

private:
  struct PoolInfo {
    hsa_amd_memory_pool_t pool{};
    uint32_t segment = 0;
    uint32_t global_flags = 0;
    bool runtime_alloc_allowed = false;
    uint32_t location = 0;
  };

  struct PoolList {
    std::vector<PoolInfo> pools;
  };

  static hsa_status_t collect_pool(hsa_amd_memory_pool_t pool, void *data) {
    auto *list = static_cast<PoolList *>(data);
    auto *get_info = layer().amd_memory_pool_get_info();
    if (list == nullptr || get_info == nullptr)
      return HSA_STATUS_ERROR;

    PoolInfo info{};
    info.pool = pool;
    if (get_info(pool, kHsaAmdMemoryPoolInfoSegment, &info.segment) != HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;
    (void)get_info(pool, kHsaAmdMemoryPoolInfoGlobalFlags, &info.global_flags);
    (void)get_info(pool, kHsaAmdMemoryPoolInfoRuntimeAllocAllowed, &info.runtime_alloc_allowed);
    (void)get_info(pool, kHsaAmdMemoryPoolInfoLocation, &info.location);
    list->pools.push_back(info);
    return HSA_STATUS_SUCCESS;
  }

  static bool same_kind(const PoolInfo &guest, const PoolInfo &host) {
    return guest.segment == host.segment && guest.global_flags == host.global_flags &&
           guest.runtime_alloc_allowed == host.runtime_alloc_allowed &&
           guest.location == host.location;
  }

  static std::optional<PoolInfo> find_match(const PoolInfo &guest,
                                            std::span<const PoolInfo> host_pools) {
    for (const PoolInfo &host : host_pools) {
      if (same_kind(guest, host))
        return host;
    }
    return std::nullopt;
  }

  void ensure_discovered() {
    std::lock_guard lock(mutex_);
    if (discovered_)
      return;
    discovered_ = true;

    auto *iterate_pools = layer().amd_agent_iterate_memory_pools();
    if (iterate_pools == nullptr)
      return;

    hsa_agent_t guest = AgentMapper::instance().guest_agent();
    hsa_agent_t host = AgentMapper::instance().host_for_guest();
    if (guest.handle == 0 || host.handle == 0)
      return;

    PoolList guest_pools;
    PoolList host_pools;
    hsa_status_t guest_status = iterate_pools(guest, collect_pool, &guest_pools);
    hsa_status_t host_status = iterate_pools(host, collect_pool, &host_pools);
    if (guest_status != HSA_STATUS_SUCCESS || host_status != HSA_STATUS_SUCCESS)
      return;

    for (const PoolInfo &guest_pool : guest_pools.pools) {
      std::optional<PoolInfo> host_pool = find_match(guest_pool, host_pools.pools);
      if (!host_pool)
        continue;
      guest_to_host_[guest_pool.pool.handle] = host_pool->pool.handle;
      trace_message(kLogDebug, "mapped guest pool=%llu to host pool=%llu",
                    static_cast<unsigned long long>(guest_pool.pool.handle),
                    static_cast<unsigned long long>(host_pool->pool.handle));
    }
  }

  std::mutex mutex_;
  bool discovered_ = false;
  std::unordered_map<uint64_t, uint64_t> guest_to_host_;
};

void clear_memory_pool_mapper() { MemoryPoolMapper::instance().clear(); }

hsa_status_t HSA_API rj_code_object_reader_create_from_memory(
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_memory();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(code_object, size, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr && code_object != nullptr) {
    if (!CodeObjectReaderRegistry::instance().store(
            *code_object_reader, static_cast<const uint8_t *>(code_object), size, nullptr)) {
      if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
        (void)original_destroy(*code_object_reader);
      *code_object_reader = {};
      std::fprintf(stderr, "[rocjitsu-hooks] failed to track memory-backed code-object reader\n");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    log_message(kLogDebug, "registered reader=%llu bytes=%zu",
                static_cast<unsigned long long>(code_object_reader->handle), size);
  }
  return status;
}

hsa_status_t HSA_API rj_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t *code_object_reader) {
  auto *original = layer().create_from_file();
  if (original == nullptr)
    return HSA_STATUS_ERROR;

  const hsa_status_t status = original(file, code_object_reader);
  if (status == HSA_STATUS_SUCCESS && code_object_reader != nullptr) {
    // File-backed readers do not expose stable ELF bytes through the later
    // load-agent callback. Hook the create path anyway so users get a direct
    // warning instead of an unexplained INVALID_ISA pass-through failure.
    std::fprintf(stderr,
                 "[rocjitsu-hooks] file-backed code-object reader=%llu is not translated; use "
                 "hsa_code_object_reader_create_from_memory for DBT hook translation\n",
                 static_cast<unsigned long long>(code_object_reader->handle));
  }
  return status;
}

hsa_status_t HSA_API rj_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  trace_message(kLogVerbose, "reader_destroy reader=%llu",
                static_cast<unsigned long long>(code_object_reader.handle));
  CodeObjectReaderRegistry::instance().remove(code_object_reader);

  auto *original = layer().destroy();
  if (original == nullptr)
    return HSA_STATUS_ERROR;
  hsa_status_t status = original(code_object_reader);
  trace_message(kLogVerbose, "reader_destroy status=%d", static_cast<int>(status));
  return status;
}

std::vector<hsa_agent_t> map_agent_array(const hsa_agent_t *agents, size_t count) {
  std::vector<hsa_agent_t> mapped;
  if (!agents || count == 0)
    return mapped;
  mapped.reserve(count);
  for (size_t i = 0; i < count; ++i)
    mapped.push_back(AgentMapper::instance().map(agents[i]));
  return mapped;
}

struct MappedAccessAgents {
  std::vector<hsa_agent_t> agents;
  std::vector<uint32_t> flags;
  bool changed = false;
};

MappedAccessAgents map_access_agent_array(const hsa_agent_t *agents, uint32_t count,
                                          const uint32_t *flags) {
  MappedAccessAgents mapped;
  if (!agents || count == 0)
    return mapped;

  mapped.agents.reserve(count);
  if (flags)
    mapped.flags.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    hsa_agent_t agent = AgentMapper::instance().map(agents[i]);
    mapped.changed = mapped.changed || agent.handle != agents[i].handle;

    bool duplicate = false;
    for (hsa_agent_t existing : mapped.agents) {
      if (existing.handle == agent.handle) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      // Mapping the visible guest to its host can make an all-agent list contain
      // the selected host twice. Forwarding unique agents avoids asking ROCR to
      // install duplicate access records for the same allocation.
      mapped.changed = true;
      continue;
    }

    mapped.agents.push_back(agent);
    if (flags)
      mapped.flags.push_back(flags[i]);
  }

  return mapped;
}

hsa_status_t HSA_API rj_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void *data),
                                       void *data) {
  auto *original = layer().iterate_agents();
  if (!original)
    return HSA_STATUS_ERROR;

  hsa_agent_t guest = AgentMapper::instance().guest_agent();
  hsa_agent_t host = AgentMapper::instance().host_for_guest();
  if (guest.handle == 0 || host.handle == 0)
    return original(callback, data);

  struct ShadowIteration {
    hsa_status_t (*callback)(hsa_agent_t, void *) = nullptr;
    void *data = nullptr;
    hsa_agent_t guest{};
    hsa_agent_t host{};
    bool guest_emitted = false;
  } shadow{callback, data, guest, host, false};

  auto shadow_callback = [](hsa_agent_t agent, void *opaque) -> hsa_status_t {
    auto *shadow = static_cast<ShadowIteration *>(opaque);
    if (agent.handle == shadow->host.handle) {
      // Public enumeration is the replacement boundary: applications see the
      // guest agent in the selected host's ordinal slot, while ROCR keeps the
      // host agent alive for translated execution.
      shadow->guest_emitted = true;
      return shadow->callback(shadow->guest, shadow->data);
    }
    if (agent.handle == shadow->guest.handle && shadow->guest_emitted) {
      // The synthetic KFD node is appended after the real GPUs. Skip that later
      // duplicate so public clients see one GPU replacing the selected host.
      return HSA_STATUS_SUCCESS;
    }
    return shadow->callback(agent, shadow->data);
  };

  trace_message(kLogDebug, "iterate_agents shadow host=%llu guest=%llu",
                static_cast<unsigned long long>(host.handle),
                static_cast<unsigned long long>(guest.handle));
  return original(shadow_callback, &shadow);
}

hsa_status_t HSA_API rj_agent_iterate_isas(hsa_agent_t agent,
                                           hsa_status_t (*callback)(hsa_isa_t isa, void *data),
                                           void *data) {
  auto *original = layer().agent_iterate_isas();
  if (!original)
    return HSA_STATUS_ERROR;

  // HIP/CLR derives the application-visible device ISA from this callback.
  // Keep the synthetic agent guest-facing so fatbin selection picks gfx950 code
  // objects; execution-facing hooks translate and map those loads to the host.
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogDebug, "agent_iterate_isas agent=%llu mapped=%llu",
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle));
  return original(agent, callback, data);
}

hsa_status_t HSA_API rj_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                     void (*callback)(hsa_status_t, hsa_queue_t *, void *),
                                     void *data, uint32_t private_segment_size,
                                     uint32_t group_segment_size, hsa_queue_t **queue) {
  auto *original = layer().queue_create();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogVerbose, "queue_create agent=%llu mapped=%llu size=%u",
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle), size);
  return original(mapped, size, type, callback, data, private_segment_size, group_segment_size,
                  queue);
}

hsa_status_t HSA_API rj_queue_destroy(hsa_queue_t *queue) {
  auto *original = layer().queue_destroy();
  if (!original)
    return HSA_STATUS_ERROR;
  trace_message(kLogVerbose, "queue_destroy queue=%p id=%llu", static_cast<void *>(queue),
                queue ? static_cast<unsigned long long>(queue->id) : 0);
  hsa_status_t status = original(queue);
  trace_message(kLogVerbose, "queue_destroy status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_agent_iterate_regions(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void *data), void *data) {
  auto *original = layer().agent_iterate_regions();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogDebug, "agent_iterate_regions agent=%llu mapped=%llu",
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle));
  return original(mapped, callback, data);
}

hsa_status_t HSA_API rj_memory_assign_agent(void *ptr, hsa_agent_t agent,
                                            hsa_access_permission_t access) {
  auto *original = layer().memory_assign_agent();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogVerbose, "memory_assign_agent ptr=%p agent=%llu mapped=%llu access=%d", ptr,
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle), static_cast<int>(access));
  return original(ptr, mapped, access);
}

hsa_status_t HSA_API rj_shut_down() {
  auto config = layer().config();
  if (config && config->guest_target) {
    // ROCR teardown currently corrupts heap state after a synthetic guest agent
    // has been initialized with host-backed resources. rocjitsu is a
    // process-scoped launcher for this MVP, so let process exit reclaim ROCR.
    trace_message(kLogVerbose, "skipping real hsa_shut_down in guest mode");
    layer().uninstall();
    return HSA_STATUS_SUCCESS;
  }

  auto *original = layer().shut_down();
  if (!original)
    return HSA_STATUS_ERROR;
  return original();
}

hsa_status_t HSA_API rj_executable_destroy(hsa_executable_t executable) {
  trace_message(kLogVerbose, "executable_destroy exec=%llu",
                static_cast<unsigned long long>(executable.handle));
  ExecutableAgentRegistry::instance().erase_executable(executable);
  auto *original = layer().executable_destroy();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_status_t status = original(executable);
  trace_message(kLogVerbose, "executable_destroy status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_executable_get_symbol(hsa_executable_t executable, const char *module_name,
                                              const char *symbol_name, hsa_agent_t agent,
                                              int32_t call_convention,
                                              hsa_executable_symbol_t *symbol) {
  auto *original = layer().executable_get_symbol();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = ExecutableAgentRegistry::instance().map_agent(executable, agent);
  mapped = AgentMapper::instance().map(mapped);
  trace_message(kLogVerbose, "get_symbol exec=%llu agent=%llu mapped=%llu symbol=%s",
                static_cast<unsigned long long>(executable.handle),
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle), symbol_name ? symbol_name : "");
  return original(executable, module_name, symbol_name, mapped, call_convention, symbol);
}

hsa_status_t HSA_API rj_executable_get_symbol_by_name(hsa_executable_t executable,
                                                      const char *symbol_name,
                                                      const hsa_agent_t *agent,
                                                      hsa_executable_symbol_t *symbol) {
  auto *original = layer().executable_get_symbol_by_name();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_agent{};
  const hsa_agent_t *mapped_ptr = nullptr;
  if (agent != nullptr) {
    mapped_agent = ExecutableAgentRegistry::instance().map_agent(executable, *agent);
    mapped_agent = AgentMapper::instance().map(mapped_agent);
    mapped_ptr = &mapped_agent;
  }
  trace_message(kLogVerbose, "get_symbol_by_name exec=%llu agent=%llu mapped=%llu symbol=%s",
                static_cast<unsigned long long>(executable.handle),
                static_cast<unsigned long long>(agent ? agent->handle : 0),
                static_cast<unsigned long long>(mapped_agent.handle),
                symbol_name ? symbol_name : "");
  return original(executable, symbol_name, mapped_ptr, symbol);
}

hsa_status_t HSA_API rj_executable_agent_global_variable_define(hsa_executable_t executable,
                                                                hsa_agent_t agent,
                                                                const char *variable_name,
                                                                void *address) {
  auto *original = layer().executable_agent_global_variable_define();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = ExecutableAgentRegistry::instance().map_agent(executable, agent);
  mapped = AgentMapper::instance().map(mapped);
  trace_message(kLogVerbose, "global_variable_define exec=%llu agent=%llu mapped=%llu name=%s",
                static_cast<unsigned long long>(executable.handle),
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle), variable_name ? variable_name : "");
  return original(executable, mapped, variable_name, address);
}

struct IterateAgentSymbolsData {
  hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t,
                           void *) = nullptr;
  void *data = nullptr;
  hsa_agent_t guest{};
};

hsa_status_t HSA_API rj_iterate_agent_symbols_callback(hsa_executable_t executable, hsa_agent_t,
                                                       hsa_executable_symbol_t symbol, void *data) {
  auto *wrapped = static_cast<IterateAgentSymbolsData *>(data);
  return wrapped->callback(executable, wrapped->guest, symbol, wrapped->data);
}

hsa_status_t HSA_API rj_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data) {
  auto *original = layer().executable_iterate_agent_symbols();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = ExecutableAgentRegistry::instance().map_agent(executable, agent);
  mapped = AgentMapper::instance().map(mapped);
  if (mapped.handle == agent.handle)
    return original(executable, mapped, callback, data);
  IterateAgentSymbolsData wrapped{callback, data, agent};
  return original(executable, mapped, rj_iterate_agent_symbols_callback, &wrapped);
}

hsa_status_t HSA_API rj_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
  auto *original = layer().amd_agent_iterate_memory_pools();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogDebug, "amd_agent_iterate_memory_pools agent=%llu mapped=%llu",
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle));
  return original(AgentMapper::instance().is_guest(agent) ? agent : mapped, callback, data);
}

hsa_status_t HSA_API rj_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                                 hsa_amd_memory_pool_info_t attribute,
                                                 void *value) {
  auto *original = layer().amd_memory_pool_get_info();
  if (!original)
    return HSA_STATUS_ERROR;
  trace_message(kLogDebug, "amd_memory_pool_get_info pool=%llu attr=%u",
                static_cast<unsigned long long>(memory_pool.handle),
                static_cast<unsigned>(attribute));
  return original(memory_pool, attribute, value);
}

hsa_status_t HSA_API rj_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                 uint32_t flags, void **ptr) {
  auto *original = layer().amd_memory_pool_allocate();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_amd_memory_pool_t mapped_pool = MemoryPoolMapper::instance().map(memory_pool);
  trace_message(kLogVerbose, "amd_memory_pool_allocate pool=%llu mapped=%llu size=%zu flags=0x%x",
                static_cast<unsigned long long>(memory_pool.handle),
                static_cast<unsigned long long>(mapped_pool.handle), size, flags);
  hsa_status_t status = original(mapped_pool, size, flags, ptr);
  trace_message(kLogVerbose, "amd_memory_pool_allocate status=%d ptr=%p", static_cast<int>(status),
                ptr ? *ptr : nullptr);
  return status;
}

hsa_status_t HSA_API rj_amd_memory_pool_free(void *ptr) {
  auto *original = layer().amd_memory_pool_free();
  if (!original)
    return HSA_STATUS_ERROR;
  trace_message(kLogVerbose, "amd_memory_pool_free ptr=%p", ptr);
  hsa_status_t status = original(ptr);
  trace_message(kLogVerbose, "amd_memory_pool_free status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                       hsa_amd_memory_pool_t memory_pool,
                                                       hsa_amd_agent_memory_pool_info_t attribute,
                                                       void *value) {
  auto *original = layer().amd_agent_memory_pool_get_info();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  hsa_amd_memory_pool_t mapped_pool = MemoryPoolMapper::instance().map(memory_pool);
  trace_message(
      kLogDebug,
      "amd_agent_memory_pool_get_info agent=%llu mapped=%llu pool=%llu mapped=%llu "
      "attr=%u",
      static_cast<unsigned long long>(agent.handle), static_cast<unsigned long long>(mapped.handle),
      static_cast<unsigned long long>(memory_pool.handle),
      static_cast<unsigned long long>(mapped_pool.handle), static_cast<unsigned>(attribute));
  return original(mapped, mapped_pool, attribute, value);
}

hsa_status_t HSA_API rj_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t *agents,
                                                const uint32_t *flags, const void *ptr) {
  auto *original = layer().amd_agents_allow_access();
  if (!original)
    return HSA_STATUS_ERROR;
  auto mapped = map_access_agent_array(agents, num_agents, flags);
  hsa_agent_t selected_host = AgentMapper::instance().host_for_guest();
  const bool guest_all_agent_access = mapped.changed && selected_host.handle != 0 && num_agents > 1;
  const uint32_t forwarded_count =
      mapped.changed ? static_cast<uint32_t>(mapped.agents.size()) : num_agents;
  const hsa_agent_t *forwarded_agents = mapped.changed ? mapped.agents.data() : agents;
  const uint32_t *forwarded_flags = mapped.changed && flags ? mapped.flags.data() : flags;
  trace_message(
      kLogVerbose, "amd_agents_allow_access ptr=%p count=%u forwarded=%u mapped=%d guest_all=%d",
      ptr, num_agents, forwarded_count, mapped.changed ? 1 : 0, guest_all_agent_access ? 1 : 0);
  hsa_status_t status = original(forwarded_count, forwarded_agents, forwarded_flags, ptr);
  trace_message(kLogVerbose, "amd_agents_allow_access status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_amd_memory_async_copy(void *dst, hsa_agent_t dst_agent, const void *src,
                                              hsa_agent_t src_agent, size_t size,
                                              uint32_t num_dep_signals,
                                              const hsa_signal_t *dep_signals,
                                              hsa_signal_t completion_signal) {
  auto *original = layer().amd_memory_async_copy();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_dst = AgentMapper::instance().map(dst_agent);
  hsa_agent_t mapped_src = AgentMapper::instance().map(src_agent);
  trace_message(kLogVerbose,
                "amd_memory_async_copy dst_agent=%llu mapped=%llu src_agent=%llu mapped=%llu "
                "size=%zu",
                static_cast<unsigned long long>(dst_agent.handle),
                static_cast<unsigned long long>(mapped_dst.handle),
                static_cast<unsigned long long>(src_agent.handle),
                static_cast<unsigned long long>(mapped_src.handle), size);
  return original(dst, mapped_dst, src, mapped_src, size, num_dep_signals, dep_signals,
                  completion_signal);
}

hsa_status_t HSA_API rj_amd_memory_async_copy_on_engine(
    void *dst, hsa_agent_t dst_agent, const void *src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t *dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  auto *original = layer().amd_memory_async_copy_on_engine();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped_dst = AgentMapper::instance().map(dst_agent);
  hsa_agent_t mapped_src = AgentMapper::instance().map(src_agent);
  trace_message(kLogVerbose,
                "amd_memory_async_copy_on_engine dst_agent=%llu mapped=%llu src_agent=%llu "
                "mapped=%llu size=%zu engine=%u",
                static_cast<unsigned long long>(dst_agent.handle),
                static_cast<unsigned long long>(mapped_dst.handle),
                static_cast<unsigned long long>(src_agent.handle),
                static_cast<unsigned long long>(mapped_src.handle), size,
                static_cast<unsigned>(engine_id));
  return original(dst, mapped_dst, src, mapped_src, size, num_dep_signals, dep_signals,
                  completion_signal, engine_id, force_copy_on_sdma);
}

hsa_status_t HSA_API rj_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t *dst, const hsa_dim3_t *dst_offset, const hsa_pitched_ptr_t *src,
    const hsa_dim3_t *src_offset, const hsa_dim3_t *range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal) {
  auto *original = layer().amd_memory_async_copy_rect();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(copy_agent);
  trace_message(kLogVerbose, "amd_memory_async_copy_rect agent=%llu mapped=%llu dir=%u",
                static_cast<unsigned long long>(copy_agent.handle),
                static_cast<unsigned long long>(mapped.handle), static_cast<unsigned>(dir));
  return original(dst, dst_offset, src, src_offset, range, mapped, dir, num_dep_signals,
                  dep_signals, completion_signal);
}

hsa_status_t HSA_API rj_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                                      uint32_t *engine_ids_mask) {
  auto *original = layer().amd_memory_copy_engine_status();
  if (!original)
    return HSA_STATUS_ERROR;
  return original(AgentMapper::instance().map(dst_agent), AgentMapper::instance().map(src_agent),
                  engine_ids_mask);
}

hsa_status_t HSA_API rj_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
                                                             hsa_agent_t src_agent,
                                                             hsa_amd_sdma_engine_id_t *engine_id) {
  auto *original = layer().amd_memory_get_preferred_copy_engine();
  if (!original)
    return HSA_STATUS_ERROR;
  return original(AgentMapper::instance().map(dst_agent), AgentMapper::instance().map(src_agent),
                  engine_id);
}

hsa_status_t HSA_API rj_amd_memory_lock(void *host_ptr, size_t size, hsa_agent_t *agents,
                                        int num_agent, void **agent_ptr) {
  auto *original = layer().amd_memory_lock();
  if (!original)
    return HSA_STATUS_ERROR;
  auto mapped = num_agent > 0 ? map_agent_array(agents, static_cast<size_t>(num_agent))
                              : std::vector<hsa_agent_t>{};
  return original(host_ptr, size, mapped.empty() ? agents : mapped.data(), num_agent, agent_ptr);
}

hsa_status_t HSA_API rj_amd_memory_lock_to_pool(void *host_ptr, size_t size, hsa_agent_t *agents,
                                                int num_agent, hsa_amd_memory_pool_t pool,
                                                uint32_t flags, void **agent_ptr) {
  auto *original = layer().amd_memory_lock_to_pool();
  if (!original)
    return HSA_STATUS_ERROR;
  auto mapped = num_agent > 0 ? map_agent_array(agents, static_cast<size_t>(num_agent))
                              : std::vector<hsa_agent_t>{};
  hsa_amd_memory_pool_t mapped_pool = MemoryPoolMapper::instance().map(pool);
  return original(host_ptr, size, mapped.empty() ? agents : mapped.data(), num_agent, mapped_pool,
                  flags, agent_ptr);
}

hsa_status_t HSA_API rj_amd_memory_fill(void *ptr, uint32_t value, size_t count) {
  auto *original = layer().amd_memory_fill();
  if (!original)
    return HSA_STATUS_ERROR;
  trace_message(kLogVerbose, "amd_memory_fill ptr=%p value=0x%x count=%zu", ptr, value, count);
  hsa_status_t status = original(ptr, value, count);
  trace_message(kLogVerbose, "amd_memory_fill status=%d", static_cast<int>(status));
  return status;
}

hsa_status_t HSA_API rj_amd_pointer_info(const void *ptr, hsa_amd_pointer_info_t *info,
                                         void *(*alloc)(size_t), uint32_t *num_agents_accessible,
                                         hsa_agent_t **accessible) {
  auto *original = layer().amd_pointer_info();
  if (!original)
    return HSA_STATUS_ERROR;
  trace_message(kLogVerbose, "amd_pointer_info ptr=%p", ptr);
  hsa_status_t status = original(ptr, info, alloc, num_agents_accessible, accessible);
  if (status == HSA_STATUS_SUCCESS) {
    if (info != nullptr &&
        info->size >= offsetof(hsa_amd_pointer_info_t, agentOwner) + sizeof(info->agentOwner))
      info->agentOwner = AgentMapper::instance().guest_for_host(info->agentOwner);
    if (num_agents_accessible != nullptr && accessible != nullptr && *accessible != nullptr) {
      for (uint32_t i = 0; i < *num_agents_accessible; ++i)
        (*accessible)[i] = AgentMapper::instance().guest_for_host((*accessible)[i]);
    }
  }
  trace_message(kLogVerbose, "amd_pointer_info status=%d owner=%llu accessible=%u",
                static_cast<int>(status),
                static_cast<unsigned long long>(info ? info->agentOwner.handle : 0),
                num_agents_accessible ? *num_agents_accessible : 0);
  return status;
}

hsa_status_t HSA_API rj_amd_svm_prefetch_async(void *ptr, size_t size, hsa_agent_t agent,
                                               uint32_t num_dep_signals,
                                               const hsa_signal_t *dep_signals,
                                               hsa_signal_t completion_signal) {
  auto *original = layer().amd_svm_prefetch_async();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogVerbose, "amd_svm_prefetch_async ptr=%p size=%zu agent=%llu mapped=%llu", ptr,
                size, static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle));
  return original(ptr, size, mapped, num_dep_signals, dep_signals, completion_signal);
}

hsa_status_t HSA_API rj_amd_vmem_set_access(void *va, size_t size,
                                            const hsa_amd_memory_access_desc_t *desc,
                                            size_t desc_cnt) {
  auto *original = layer().amd_vmem_set_access();
  if (!original)
    return HSA_STATUS_ERROR;
  std::vector<hsa_amd_memory_access_desc_t> mapped;
  if (desc && desc_cnt > 0) {
    mapped.assign(desc, desc + desc_cnt);
    for (auto &entry : mapped)
      entry.agent_handle = AgentMapper::instance().map(entry.agent_handle);
  }
  trace_message(kLogVerbose, "amd_vmem_set_access va=%p size=%zu desc_cnt=%zu", va, size, desc_cnt);
  return original(va, size, mapped.empty() ? desc : mapped.data(), desc_cnt);
}

hsa_status_t HSA_API rj_amd_vmem_get_access(void *va, hsa_access_permission_t *perms,
                                            hsa_agent_t agent_handle) {
  auto *original = layer().amd_vmem_get_access();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent_handle);
  trace_message(kLogVerbose, "amd_vmem_get_access va=%p agent=%llu mapped=%llu", va,
                static_cast<unsigned long long>(agent_handle.handle),
                static_cast<unsigned long long>(mapped.handle));
  return original(va, perms, mapped);
}

hsa_status_t HSA_API rj_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold) {
  auto *original = layer().amd_agent_set_async_scratch_limit();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogVerbose,
                "amd_agent_set_async_scratch_limit agent=%llu mapped=%llu threshold=%zu",
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle), threshold);
  return original(mapped, threshold);
}

hsa_status_t HSA_API rj_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t *copy_ops,
                                                    uint32_t num_copy_ops, uint32_t num_dep_signals,
                                                    const hsa_signal_t *dep_signals) {
  auto *original = layer().amd_memory_async_batch_copy();
  if (!original)
    return HSA_STATUS_ERROR;
  if (!copy_ops || num_copy_ops == 0)
    return original(copy_ops, num_copy_ops, num_dep_signals, dep_signals);

  std::vector<hsa_amd_memory_copy_op_t> mapped(copy_ops, copy_ops + num_copy_ops);
  std::vector<std::vector<hsa_agent_t>> mapped_dst_lists;
  mapped_dst_lists.reserve(num_copy_ops);
  trace_message(kLogVerbose, "amd_memory_async_batch_copy ops=%u deps=%u", num_copy_ops,
                num_dep_signals);
  for (uint32_t i = 0; i < num_copy_ops; ++i) {
    auto &op = mapped[i];
    op.src_agent = AgentMapper::instance().map(op.src_agent);
    if (op.num_entries == 0) {
      op.dst_agent = AgentMapper::instance().map(op.dst_agent);
      continue;
    }
    if (op.dst_agent_list != nullptr) {
      mapped_dst_lists.push_back(map_agent_array(op.dst_agent_list, op.num_entries));
      op.dst_agent_list = mapped_dst_lists.back().data();
    } else {
      op.dst_agent = AgentMapper::instance().map(op.dst_agent);
    }
  }
  return original(mapped.data(), num_copy_ops, num_dep_signals, dep_signals);
}

hsa_status_t HSA_API rj_amd_agent_preload(hsa_agent_t agent, uint64_t flags) {
  auto *original = layer().amd_agent_preload();
  if (!original)
    return HSA_STATUS_ERROR;
  hsa_agent_t mapped = AgentMapper::instance().map(agent);
  trace_message(kLogVerbose, "amd_agent_preload agent=%llu mapped=%llu flags=0x%llx",
                static_cast<unsigned long long>(agent.handle),
                static_cast<unsigned long long>(mapped.handle),
                static_cast<unsigned long long>(flags));
  return original(mapped, flags);
}

[[nodiscard]] hsa_status_t create_translated_reader(std::vector<uint8_t> translated,
                                                    hsa_code_object_reader_t *translated_reader) {
  auto *owned = new (std::nothrow) std::vector<uint8_t>(std::move(translated));
  if (owned == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto *original_create = layer().create_from_memory();
  if (original_create == nullptr) {
    delete owned;
    return HSA_STATUS_ERROR;
  }

  const hsa_status_t status = original_create(owned->data(), owned->size(), translated_reader);
  if (status != HSA_STATUS_SUCCESS) {
    delete owned;
    return status;
  }

  if (!CodeObjectReaderRegistry::instance().store(*translated_reader, owned->data(), owned->size(),
                                                  owned)) {
    if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
      (void)original_destroy(*translated_reader);
    *translated_reader = {};
    delete owned;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API rj_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char *options, hsa_loaded_code_object_t *loaded_code_object) {
  auto *original_load = layer().load_agent_code_object();
  if (original_load == nullptr)
    return HSA_STATUS_ERROR;

  auto config = layer().config();
  if (!config) {
    // `OnUnload()` should not normally race active ROCR callbacks, but returning
    // an HSA error here is safer than dereferencing cleared hook state.
    std::fprintf(stderr, "[rocjitsu-hooks] DBT hook layer is inactive during code-object load\n");
    return HSA_STATUS_ERROR;
  }

  const bool guest_load = AgentMapper::instance().is_guest(agent);
  trace_message(kLogVerbose, "load_agent_code_object exec=%llu agent=%llu guest=%d reader=%llu",
                static_cast<unsigned long long>(executable.handle),
                static_cast<unsigned long long>(agent.handle), guest_load ? 1 : 0,
                static_cast<unsigned long long>(code_object_reader.handle));
  if (config->guest_target && !guest_load) {
    hsa_status_t status =
        original_load(executable, agent, code_object_reader, options, loaded_code_object);
    trace_message(kLogVerbose, "load_agent_code_object host/pass-through status=%d",
                  static_cast<int>(status));
    return status;
  }

  hsa_agent_t load_agent = guest_load ? AgentMapper::instance().host_for_guest() : agent;
  if (guest_load && load_agent.handle == 0)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  const uint8_t *bytes = nullptr;
  size_t size = 0;
  if (!CodeObjectReaderRegistry::instance().lookup(code_object_reader, &bytes, &size)) {
    log_message(kLogInfo, "no memory bytes registered for reader=%llu",
                static_cast<unsigned long long>(code_object_reader.handle));
    if (guest_load)
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
    return original_load(executable, load_agent, code_object_reader, options, loaded_code_object);
  }

  const DetectedElfTarget detected = detect_target_from_elf(bytes, size);
  DetectedElfTarget source_target = detected;
  const bool detected_already_target =
      detected.arch == config->target.arch && detected.mach == config->target.mach;
  if (config->source_override && !detected_already_target) {
    source_target.arch = config->source_override->arch;
    source_target.mach = config->source_override->mach;
  }
  if (source_target.arch == ROCJITSU_CODE_ARCH_INVALID || source_target.mach == 0) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to detect source ISA from code-object ELF\n");
    return HSA_STATUS_ERROR;
  }

  if (source_target.arch == config->target.arch && source_target.mach == config->target.mach) {
    log_message(kLogInfo,
                "source target %s arch %s already matches requested target; passing through",
                elf_mach_name(source_target.mach), arch_name(source_target.arch));
    hsa_status_t status =
        original_load(executable, load_agent, code_object_reader, options, loaded_code_object);
    trace_message(kLogVerbose, "load_agent_code_object already-target status=%d",
                  static_cast<int>(status));
    if (status == HSA_STATUS_SUCCESS && guest_load)
      ExecutableAgentRegistry::instance().record(executable, agent, load_agent);
    return status;
  }

  log_message(kLogInfo, "translating reader=%llu %s/%s -> %s/%s mach=0x%x",
              static_cast<unsigned long long>(code_object_reader.handle),
              elf_mach_name(source_target.mach), arch_name(source_target.arch),
              config->target.name.data(), arch_name(config->target.arch), config->target.mach);

  AmdGpuCodeObject source_object(bytes, size);
  if (!source_object.is_valid()) {
    std::fprintf(stderr, "[rocjitsu-hooks] source bytes are not a valid AMDGPU code object\n");
    return HSA_STATUS_ERROR;
  }

  rocjitsu::TranslatedCodeObject translated;
  BinaryTranslatorOptions translator_options;
  BinaryTranslator translator(source_target.arch, config->target.arch, config->target.mach,
                              translator_options);
  translated = translator.translate(source_object);

  print_diagnostics(stderr, translated.diagnostics, config->log_level > kLogDisabled);
  if (translated.elf_bytes.empty() || has_error_diagnostic(translated.diagnostics)) {
    std::fprintf(stderr, "[rocjitsu-hooks] translation failed; refusing original code object\n");
    return HSA_STATUS_ERROR;
  }

  hsa_code_object_reader_t translated_reader{};
  hsa_status_t status =
      create_translated_reader(std::move(translated.elf_bytes), &translated_reader);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to create translated code-object reader: %d\n",
                 static_cast<int>(status));
    return status;
  }

  status = original_load(executable, load_agent, translated_reader, options, loaded_code_object);
  trace_message(kLogVerbose, "load_agent_code_object translated load_agent=%llu status=%d",
                static_cast<unsigned long long>(load_agent.handle), static_cast<int>(status));
  CodeObjectReaderRegistry::instance().remove(translated_reader);
  if (auto *original_destroy = layer().destroy(); original_destroy != nullptr)
    (void)original_destroy(translated_reader);

  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "[rocjitsu-hooks] translated code-object load failed: %d\n",
                 static_cast<int>(status));
  } else if (guest_load) {
    ExecutableAgentRegistry::instance().record(executable, agent, load_agent);
  }
  return status;
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RJ_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define RJ_HOOK_EXPORT
#endif

/// @brief ROCR HSA tools entry point.
///
/// @details Saves the incoming `CoreApiTable` function pointers and installs
/// DBT load-time wrappers when the `RJ_CONFIG` file enables `dbt_guest`. The
/// failed tool list is not modified; ROCR owns that state and passes it for
/// diagnostics only.
extern "C" RJ_HOOK_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                      uint64_t failed_tool_count,
                                      const char *const *failed_tool_names) {
  (void)runtime_version;
  (void)failed_tool_count;
  (void)failed_tool_names;

  auto config = parse_config();
  if (!config)
    return false;
  maybe_install_signal_backtrace(config->signal_backtrace);
  return layer().install(table, std::move(*config));
}

/// @brief ROCR HSA tools unload entry point.
///
/// @details Restores rocjitsu wrappers that are still installed and clears
/// process-local reader state owned by the hook. ROCR also resets the API table
/// after unloading tools, but the hook does its own cleanup so tests and future
/// in-process reload paths do not retain stale translated ELF buffers.
extern "C" RJ_HOOK_EXPORT void OnUnload() { layer().uninstall(); }
