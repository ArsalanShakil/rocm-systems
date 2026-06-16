// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_api_trace_minimal.h
/// @brief Minimal ROCR HSA tools API-table ABI mirror used by rocjitsu hooks.
///
/// @details ROCR passes HSA tools a table whose layout is defined by
/// `hsa_api_trace.h`. The full ROCR trace header pulls in extension headers that
/// are unnecessary for the DBT-only MVP. This file mirrors only the table
/// headers and the core API entries through the code-object load functions that
/// rocjitsu patches. Field order must match ROCR exactly for every mirrored
/// field; callers must validate `CoreApiTable::version.minor_id` before reading
/// or writing function pointers near the tail of the table.

#pragma once

#include "hsa/hsa.h"

#include <cstddef>
#include <cstdint>

/// @brief Version header present at the start of every ROCR API table.
///
/// @details ROCR stores the table size in `minor_id` for the HSA tools ABI. The
/// DBT hook uses that value as an ABI boundary check before dereferencing fields
/// that may not exist in older runtimes.
struct ApiTableVersion {
  uint32_t major_id;
  uint32_t minor_id;
  uint32_t step_id;
  uint32_t reserved;
};

/// @brief Minimal AMD extension memory-pool handle mirror.
struct hsa_amd_memory_pool_s {
  uint64_t handle;
};
using hsa_amd_memory_pool_t = hsa_amd_memory_pool_s;

using hsa_amd_memory_pool_info_t = uint32_t;
using hsa_amd_agent_memory_pool_info_t = uint32_t;
using hsa_amd_sdma_engine_id_t = uint32_t;
using hsa_amd_pointer_type_t = uint32_t;
using hsa_amd_copy_direction_t = uint32_t;

/// @brief Minimal pitched pointer mirror for hsa_amd_memory_async_copy_rect.
struct hsa_pitched_ptr_t {
  void *base;
  size_t pitch;
  size_t slice;
};

/// @brief Minimal AMD virtual-memory access descriptor mirror.
struct hsa_amd_memory_access_desc_t {
  hsa_access_permission_t permissions;
  hsa_agent_t agent_handle;
};

/// @brief Minimal AMD pointer info mirror used to rewrite app-facing agent fields.
struct hsa_amd_pointer_info_t {
  uint32_t size;
  hsa_amd_pointer_type_t type;
  void *agentBaseAddress;
  void *hostBaseAddress;
  size_t sizeInBytes;
  void *userData;
  hsa_agent_t agentOwner;
  uint32_t global_flags;
  bool registered;
};

/// @brief Operation type values used by hsa_amd_memory_async_batch_copy.
enum hsa_amd_memory_copy_op_type_t : uint16_t {
  HSA_AMD_MEMORY_COPY_OP_LINEAR = 0,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST = 1,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP = 2,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC = 3,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST = 4,
  HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST = 5,
};

/// @brief Minimal layout mirror of ROCR's batch-copy operation descriptor.
///
/// @details The HSA hook copies these descriptors only to rewrite agent
/// handles from guest to host before forwarding the call. Field order and
/// widths must match hsa_ext_amd.h.
struct hsa_amd_memory_copy_op_t {
  uint16_t version;
  uint16_t type;
  uint16_t num_entries;
  uint16_t traffic_class;
  hsa_signal_t completion_signal;
  union {
    void *src;
    void **src_list;
  };
  union {
    hsa_agent_t src_agent;
    hsa_agent_t *src_agent_list;
  };
  union {
    hsa_agent_t dst_agent;
    hsa_agent_t *dst_agent_list;
  };
  union {
    void *dst;
    void **dst_list;
  };
  union {
    struct {
      size_t size;
      size_t unused_size;
    };
    struct {
      size_t src_size;
      size_t dst_size;
    };
    struct {
      size_t *size_list;
      size_t reserved0;
    };
  };
  struct {
    uint16_t function;
    uint16_t scope;
    uint32_t reserved;
    void *addr;
    uint64_t value;
    uint64_t mask;
  } wait;
  struct {
    uint16_t operation;
    uint16_t scope;
    uint32_t reserved;
    void *addr;
    uint64_t data;
  } signal;
  uint64_t reserved1[1];
};

using hsa_amd_agent_iterate_memory_pools_fn_t = hsa_status_t(HSA_API *)(
    hsa_agent_t, hsa_status_t (*)(hsa_amd_memory_pool_t memory_pool, void *data), void *);
using hsa_amd_memory_pool_get_info_fn_t = hsa_status_t(HSA_API *)(hsa_amd_memory_pool_t,
                                                                  hsa_amd_memory_pool_info_t,
                                                                  void *);
using hsa_amd_agent_memory_pool_get_info_fn_t = hsa_status_t(HSA_API *)(
    hsa_agent_t, hsa_amd_memory_pool_t, hsa_amd_agent_memory_pool_info_t, void *);
using hsa_amd_memory_pool_allocate_fn_t = hsa_status_t(HSA_API *)(hsa_amd_memory_pool_t, size_t,
                                                                  uint32_t, void **);
using hsa_amd_memory_pool_free_fn_t = hsa_status_t(HSA_API *)(void *);
using hsa_amd_agents_allow_access_fn_t = hsa_status_t(HSA_API *)(uint32_t, const hsa_agent_t *,
                                                                 const uint32_t *, const void *);
using hsa_amd_memory_async_copy_fn_t = hsa_status_t(HSA_API *)(void *, hsa_agent_t, const void *,
                                                               hsa_agent_t, size_t, uint32_t,
                                                               const hsa_signal_t *, hsa_signal_t);
using hsa_amd_memory_async_copy_on_engine_fn_t =
    hsa_status_t(HSA_API *)(void *, hsa_agent_t, const void *, hsa_agent_t, size_t, uint32_t,
                            const hsa_signal_t *, hsa_signal_t, hsa_amd_sdma_engine_id_t, bool);
using hsa_amd_memory_copy_engine_status_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t, hsa_agent_t,
                                                                       uint32_t *);
using hsa_amd_memory_get_preferred_copy_engine_fn_t =
    hsa_status_t(HSA_API *)(hsa_agent_t, hsa_agent_t, hsa_amd_sdma_engine_id_t *);
using hsa_amd_memory_lock_fn_t = hsa_status_t(HSA_API *)(void *, size_t, hsa_agent_t *, int,
                                                         void **);
using hsa_amd_memory_fill_fn_t = hsa_status_t(HSA_API *)(void *, uint32_t, size_t);
using hsa_amd_pointer_info_fn_t = hsa_status_t(HSA_API *)(const void *, hsa_amd_pointer_info_t *,
                                                          void *(*)(size_t), uint32_t *,
                                                          hsa_agent_t **);
using hsa_amd_memory_lock_to_pool_fn_t = hsa_status_t(HSA_API *)(void *, size_t, hsa_agent_t *, int,
                                                                 hsa_amd_memory_pool_t, uint32_t,
                                                                 void **);
using hsa_amd_svm_prefetch_async_fn_t = hsa_status_t(HSA_API *)(void *, size_t, hsa_agent_t,
                                                                uint32_t, const hsa_signal_t *,
                                                                hsa_signal_t);
using hsa_amd_vmem_set_access_fn_t = hsa_status_t(HSA_API *)(void *, size_t,
                                                             const hsa_amd_memory_access_desc_t *,
                                                             size_t);
using hsa_amd_vmem_get_access_fn_t = hsa_status_t(HSA_API *)(void *, hsa_access_permission_t *,
                                                             hsa_agent_t);
using hsa_amd_memory_async_copy_rect_fn_t = hsa_status_t(HSA_API *)(
    const hsa_pitched_ptr_t *, const hsa_dim3_t *, const hsa_pitched_ptr_t *, const hsa_dim3_t *,
    const hsa_dim3_t *, hsa_agent_t, hsa_amd_copy_direction_t, uint32_t, const hsa_signal_t *,
    hsa_signal_t);
using hsa_amd_agent_set_async_scratch_limit_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t, size_t);
using hsa_amd_memory_async_batch_copy_fn_t = hsa_status_t(HSA_API *)(
    const hsa_amd_memory_copy_op_t *, uint32_t, uint32_t, const hsa_signal_t *);
using hsa_amd_agent_preload_fn_t = hsa_status_t(HSA_API *)(hsa_agent_t, uint64_t);

/// @brief Minimal mirror of ROCR's AMD extension API table.
///
/// @details Only patched entries are typed. Other fields remain opaque but
/// preserve their exact positions in the table.
struct AmdExtTable {
  ApiTableVersion version;
  void *hsa_amd_coherency_get_type_fn;
  void *hsa_amd_coherency_set_type_fn;
  void *hsa_amd_profiling_set_profiler_enabled_fn;
  void *hsa_amd_profiling_async_copy_enable_fn;
  void *hsa_amd_profiling_get_dispatch_time_fn;
  void *hsa_amd_profiling_get_async_copy_time_fn;
  void *hsa_amd_profiling_convert_tick_to_system_domain_fn;
  void *hsa_amd_signal_async_handler_fn;
  void *hsa_amd_async_function_fn;
  void *hsa_amd_signal_wait_any_fn;
  void *hsa_amd_queue_cu_set_mask_fn;
  hsa_amd_memory_pool_get_info_fn_t hsa_amd_memory_pool_get_info_fn;
  hsa_amd_agent_iterate_memory_pools_fn_t hsa_amd_agent_iterate_memory_pools_fn;
  hsa_amd_memory_pool_allocate_fn_t hsa_amd_memory_pool_allocate_fn;
  hsa_amd_memory_pool_free_fn_t hsa_amd_memory_pool_free_fn;
  hsa_amd_memory_async_copy_fn_t hsa_amd_memory_async_copy_fn;
  hsa_amd_memory_async_copy_on_engine_fn_t hsa_amd_memory_async_copy_on_engine_fn;
  hsa_amd_memory_copy_engine_status_fn_t hsa_amd_memory_copy_engine_status_fn;
  hsa_amd_agent_memory_pool_get_info_fn_t hsa_amd_agent_memory_pool_get_info_fn;
  hsa_amd_agents_allow_access_fn_t hsa_amd_agents_allow_access_fn;
  void *hsa_amd_memory_pool_can_migrate_fn;
  void *hsa_amd_memory_migrate_fn;
  hsa_amd_memory_lock_fn_t hsa_amd_memory_lock_fn;
  void *hsa_amd_memory_unlock_fn;
  hsa_amd_memory_fill_fn_t hsa_amd_memory_fill_fn;
  void *hsa_amd_interop_map_buffer_fn;
  void *hsa_amd_interop_unmap_buffer_fn;
  void *hsa_amd_image_create_fn;
  hsa_amd_pointer_info_fn_t hsa_amd_pointer_info_fn;
  void *hsa_amd_pointer_info_set_userdata_fn;
  void *hsa_amd_ipc_memory_create_fn;
  void *hsa_amd_ipc_memory_attach_fn;
  void *hsa_amd_ipc_memory_detach_fn;
  void *hsa_amd_signal_create_fn;
  void *hsa_amd_ipc_signal_create_fn;
  void *hsa_amd_ipc_signal_attach_fn;
  void *hsa_amd_register_system_event_handler_fn;
  void *hsa_amd_queue_intercept_create_fn;
  void *hsa_amd_queue_intercept_register_fn;
  void *hsa_amd_queue_set_priority_fn;
  hsa_amd_memory_async_copy_rect_fn_t hsa_amd_memory_async_copy_rect_fn;
  void *hsa_amd_runtime_queue_create_register_fn;
  hsa_amd_memory_lock_to_pool_fn_t hsa_amd_memory_lock_to_pool_fn;
  void *hsa_amd_register_deallocation_callback_fn;
  void *hsa_amd_deregister_deallocation_callback_fn;
  void *hsa_amd_signal_value_pointer_fn;
  void *hsa_amd_svm_attributes_set_fn;
  void *hsa_amd_svm_attributes_get_fn;
  hsa_amd_svm_prefetch_async_fn_t hsa_amd_svm_prefetch_async_fn;
  void *hsa_amd_spm_acquire_fn;
  void *hsa_amd_spm_release_fn;
  void *hsa_amd_spm_set_dest_buffer_fn;
  void *hsa_amd_queue_cu_get_mask_fn;
  void *hsa_amd_portable_export_dmabuf_fn;
  void *hsa_amd_portable_close_dmabuf_fn;
  void *hsa_amd_vmem_address_reserve_fn;
  void *hsa_amd_vmem_address_free_fn;
  void *hsa_amd_vmem_handle_create_fn;
  void *hsa_amd_vmem_handle_release_fn;
  void *hsa_amd_vmem_map_fn;
  void *hsa_amd_vmem_unmap_fn;
  hsa_amd_vmem_set_access_fn_t hsa_amd_vmem_set_access_fn;
  hsa_amd_vmem_get_access_fn_t hsa_amd_vmem_get_access_fn;
  void *hsa_amd_vmem_export_shareable_handle_fn;
  void *hsa_amd_vmem_import_shareable_handle_fn;
  void *hsa_amd_vmem_retain_alloc_handle_fn;
  void *hsa_amd_vmem_get_alloc_properties_from_handle_fn;
  hsa_amd_agent_set_async_scratch_limit_fn_t hsa_amd_agent_set_async_scratch_limit_fn;
  void *hsa_amd_queue_get_info_fn;
  void *hsa_amd_vmem_address_reserve_align_fn;
  void *hsa_amd_enable_logging_fn;
  void *hsa_amd_signal_wait_all_fn;
  hsa_amd_memory_get_preferred_copy_engine_fn_t hsa_amd_memory_get_preferred_copy_engine_fn;
  void *hsa_amd_portable_export_dmabuf_v2_fn;
  void *hsa_amd_ais_file_write_fn;
  void *hsa_amd_ais_file_read_fn;
  void *hsa_amd_counted_queue_acquire_fn;
  void *hsa_amd_counted_queue_release_fn;
  hsa_amd_memory_async_batch_copy_fn_t hsa_amd_memory_async_batch_copy_fn;
  hsa_amd_agent_preload_fn_t hsa_amd_agent_preload_fn;
  void *hsa_amd_svm_discard_batch_async_fn;
  void *hsa_amd_signal_get_event_id_fn;
  void *hsa_amd_external_semaphore_handle_open_fn;
  void *hsa_amd_external_semaphore_handle_close_fn;
};

/// @brief Minimal mirror of ROCR's `CoreApiTable` through agent code-object load.
///
/// @details The field order and types intentionally match ROCR
/// `hsa_api_trace.h` through `hsa_executable_load_agent_code_object_fn`.
/// Rocjitsu overwrites the file-reader create function for diagnostics, the
/// memory-reader create/destroy functions for translation state, and the agent
/// code-object loader. Preserving the preceding fields keeps the offsets
/// identical to ROCR without vendoring extension-table dependencies.
struct CoreApiTable {
  ApiTableVersion version;
  decltype(hsa_init) *hsa_init_fn;
  decltype(hsa_shut_down) *hsa_shut_down_fn;
  decltype(hsa_system_get_info) *hsa_system_get_info_fn;
  decltype(hsa_system_extension_supported) *hsa_system_extension_supported_fn;
  decltype(hsa_system_get_extension_table) *hsa_system_get_extension_table_fn;
  decltype(hsa_iterate_agents) *hsa_iterate_agents_fn;
  decltype(hsa_agent_get_info) *hsa_agent_get_info_fn;
  decltype(hsa_queue_create) *hsa_queue_create_fn;
  decltype(hsa_soft_queue_create) *hsa_soft_queue_create_fn;
  decltype(hsa_queue_destroy) *hsa_queue_destroy_fn;
  decltype(hsa_queue_inactivate) *hsa_queue_inactivate_fn;
  decltype(hsa_queue_load_read_index_scacquire) *hsa_queue_load_read_index_scacquire_fn;
  decltype(hsa_queue_load_read_index_relaxed) *hsa_queue_load_read_index_relaxed_fn;
  decltype(hsa_queue_load_write_index_scacquire) *hsa_queue_load_write_index_scacquire_fn;
  decltype(hsa_queue_load_write_index_relaxed) *hsa_queue_load_write_index_relaxed_fn;
  decltype(hsa_queue_store_write_index_relaxed) *hsa_queue_store_write_index_relaxed_fn;
  decltype(hsa_queue_store_write_index_screlease) *hsa_queue_store_write_index_screlease_fn;
  decltype(hsa_queue_cas_write_index_scacq_screl) *hsa_queue_cas_write_index_scacq_screl_fn;
  decltype(hsa_queue_cas_write_index_scacquire) *hsa_queue_cas_write_index_scacquire_fn;
  decltype(hsa_queue_cas_write_index_relaxed) *hsa_queue_cas_write_index_relaxed_fn;
  decltype(hsa_queue_cas_write_index_screlease) *hsa_queue_cas_write_index_screlease_fn;
  decltype(hsa_queue_add_write_index_scacq_screl) *hsa_queue_add_write_index_scacq_screl_fn;
  decltype(hsa_queue_add_write_index_scacquire) *hsa_queue_add_write_index_scacquire_fn;
  decltype(hsa_queue_add_write_index_relaxed) *hsa_queue_add_write_index_relaxed_fn;
  decltype(hsa_queue_add_write_index_screlease) *hsa_queue_add_write_index_screlease_fn;
  decltype(hsa_queue_store_read_index_relaxed) *hsa_queue_store_read_index_relaxed_fn;
  decltype(hsa_queue_store_read_index_screlease) *hsa_queue_store_read_index_screlease_fn;
  decltype(hsa_agent_iterate_regions) *hsa_agent_iterate_regions_fn;
  decltype(hsa_region_get_info) *hsa_region_get_info_fn;
  decltype(hsa_agent_get_exception_policies) *hsa_agent_get_exception_policies_fn;
  decltype(hsa_agent_extension_supported) *hsa_agent_extension_supported_fn;
  decltype(hsa_memory_register) *hsa_memory_register_fn;
  decltype(hsa_memory_deregister) *hsa_memory_deregister_fn;
  decltype(hsa_memory_allocate) *hsa_memory_allocate_fn;
  decltype(hsa_memory_free) *hsa_memory_free_fn;
  decltype(hsa_memory_copy) *hsa_memory_copy_fn;
  decltype(hsa_memory_assign_agent) *hsa_memory_assign_agent_fn;
  decltype(hsa_signal_create) *hsa_signal_create_fn;
  decltype(hsa_signal_destroy) *hsa_signal_destroy_fn;
  decltype(hsa_signal_load_relaxed) *hsa_signal_load_relaxed_fn;
  decltype(hsa_signal_load_scacquire) *hsa_signal_load_scacquire_fn;
  decltype(hsa_signal_store_relaxed) *hsa_signal_store_relaxed_fn;
  decltype(hsa_signal_store_screlease) *hsa_signal_store_screlease_fn;
  decltype(hsa_signal_wait_relaxed) *hsa_signal_wait_relaxed_fn;
  decltype(hsa_signal_wait_scacquire) *hsa_signal_wait_scacquire_fn;
  decltype(hsa_signal_and_relaxed) *hsa_signal_and_relaxed_fn;
  decltype(hsa_signal_and_scacquire) *hsa_signal_and_scacquire_fn;
  decltype(hsa_signal_and_screlease) *hsa_signal_and_screlease_fn;
  decltype(hsa_signal_and_scacq_screl) *hsa_signal_and_scacq_screl_fn;
  decltype(hsa_signal_or_relaxed) *hsa_signal_or_relaxed_fn;
  decltype(hsa_signal_or_scacquire) *hsa_signal_or_scacquire_fn;
  decltype(hsa_signal_or_screlease) *hsa_signal_or_screlease_fn;
  decltype(hsa_signal_or_scacq_screl) *hsa_signal_or_scacq_screl_fn;
  decltype(hsa_signal_xor_relaxed) *hsa_signal_xor_relaxed_fn;
  decltype(hsa_signal_xor_scacquire) *hsa_signal_xor_scacquire_fn;
  decltype(hsa_signal_xor_screlease) *hsa_signal_xor_screlease_fn;
  decltype(hsa_signal_xor_scacq_screl) *hsa_signal_xor_scacq_screl_fn;
  decltype(hsa_signal_exchange_relaxed) *hsa_signal_exchange_relaxed_fn;
  decltype(hsa_signal_exchange_scacquire) *hsa_signal_exchange_scacquire_fn;
  decltype(hsa_signal_exchange_screlease) *hsa_signal_exchange_screlease_fn;
  decltype(hsa_signal_exchange_scacq_screl) *hsa_signal_exchange_scacq_screl_fn;
  decltype(hsa_signal_add_relaxed) *hsa_signal_add_relaxed_fn;
  decltype(hsa_signal_add_scacquire) *hsa_signal_add_scacquire_fn;
  decltype(hsa_signal_add_screlease) *hsa_signal_add_screlease_fn;
  decltype(hsa_signal_add_scacq_screl) *hsa_signal_add_scacq_screl_fn;
  decltype(hsa_signal_subtract_relaxed) *hsa_signal_subtract_relaxed_fn;
  decltype(hsa_signal_subtract_scacquire) *hsa_signal_subtract_scacquire_fn;
  decltype(hsa_signal_subtract_screlease) *hsa_signal_subtract_screlease_fn;
  decltype(hsa_signal_subtract_scacq_screl) *hsa_signal_subtract_scacq_screl_fn;
  decltype(hsa_signal_cas_relaxed) *hsa_signal_cas_relaxed_fn;
  decltype(hsa_signal_cas_scacquire) *hsa_signal_cas_scacquire_fn;
  decltype(hsa_signal_cas_screlease) *hsa_signal_cas_screlease_fn;
  decltype(hsa_signal_cas_scacq_screl) *hsa_signal_cas_scacq_screl_fn;
  decltype(hsa_isa_from_name) *hsa_isa_from_name_fn;
  decltype(hsa_isa_get_info) *hsa_isa_get_info_fn;
  decltype(hsa_isa_compatible) *hsa_isa_compatible_fn;
  decltype(hsa_code_object_serialize) *hsa_code_object_serialize_fn;
  decltype(hsa_code_object_deserialize) *hsa_code_object_deserialize_fn;
  decltype(hsa_code_object_destroy) *hsa_code_object_destroy_fn;
  decltype(hsa_code_object_get_info) *hsa_code_object_get_info_fn;
  decltype(hsa_code_object_get_symbol) *hsa_code_object_get_symbol_fn;
  decltype(hsa_code_symbol_get_info) *hsa_code_symbol_get_info_fn;
  decltype(hsa_code_object_iterate_symbols) *hsa_code_object_iterate_symbols_fn;
  decltype(hsa_executable_create) *hsa_executable_create_fn;
  decltype(hsa_executable_destroy) *hsa_executable_destroy_fn;
  decltype(hsa_executable_load_code_object) *hsa_executable_load_code_object_fn;
  decltype(hsa_executable_freeze) *hsa_executable_freeze_fn;
  decltype(hsa_executable_get_info) *hsa_executable_get_info_fn;
  decltype(hsa_executable_global_variable_define) *hsa_executable_global_variable_define_fn;
  decltype(hsa_executable_agent_global_variable_define)
      *hsa_executable_agent_global_variable_define_fn;
  decltype(hsa_executable_readonly_variable_define) *hsa_executable_readonly_variable_define_fn;
  decltype(hsa_executable_validate) *hsa_executable_validate_fn;
  decltype(hsa_executable_get_symbol) *hsa_executable_get_symbol_fn;
  decltype(hsa_executable_symbol_get_info) *hsa_executable_symbol_get_info_fn;
  decltype(hsa_executable_iterate_symbols) *hsa_executable_iterate_symbols_fn;
  decltype(hsa_status_string) *hsa_status_string_fn;
  decltype(hsa_extension_get_name) *hsa_extension_get_name_fn;
  decltype(hsa_system_major_extension_supported) *hsa_system_major_extension_supported_fn;
  decltype(hsa_system_get_major_extension_table) *hsa_system_get_major_extension_table_fn;
  decltype(hsa_agent_major_extension_supported) *hsa_agent_major_extension_supported_fn;
  decltype(hsa_cache_get_info) *hsa_cache_get_info_fn;
  decltype(hsa_agent_iterate_caches) *hsa_agent_iterate_caches_fn;
  decltype(hsa_signal_silent_store_relaxed) *hsa_signal_silent_store_relaxed_fn;
  decltype(hsa_signal_silent_store_screlease) *hsa_signal_silent_store_screlease_fn;
  decltype(hsa_signal_group_create) *hsa_signal_group_create_fn;
  decltype(hsa_signal_group_destroy) *hsa_signal_group_destroy_fn;
  decltype(hsa_signal_group_wait_any_scacquire) *hsa_signal_group_wait_any_scacquire_fn;
  decltype(hsa_signal_group_wait_any_relaxed) *hsa_signal_group_wait_any_relaxed_fn;
  decltype(hsa_agent_iterate_isas) *hsa_agent_iterate_isas_fn;
  decltype(hsa_isa_get_info_alt) *hsa_isa_get_info_alt_fn;
  decltype(hsa_isa_get_exception_policies) *hsa_isa_get_exception_policies_fn;
  decltype(hsa_isa_get_round_method) *hsa_isa_get_round_method_fn;
  decltype(hsa_wavefront_get_info) *hsa_wavefront_get_info_fn;
  decltype(hsa_isa_iterate_wavefronts) *hsa_isa_iterate_wavefronts_fn;
  decltype(hsa_code_object_get_symbol_from_name) *hsa_code_object_get_symbol_from_name_fn;
  decltype(hsa_code_object_reader_create_from_file) *hsa_code_object_reader_create_from_file_fn;
  decltype(hsa_code_object_reader_create_from_memory) *hsa_code_object_reader_create_from_memory_fn;
  decltype(hsa_code_object_reader_destroy) *hsa_code_object_reader_destroy_fn;
  decltype(hsa_executable_create_alt) *hsa_executable_create_alt_fn;
  decltype(hsa_executable_load_program_code_object) *hsa_executable_load_program_code_object_fn;
  decltype(hsa_executable_load_agent_code_object) *hsa_executable_load_agent_code_object_fn;
  decltype(hsa_executable_validate_alt) *hsa_executable_validate_alt_fn;
  decltype(hsa_executable_get_symbol_by_name) *hsa_executable_get_symbol_by_name_fn;
  decltype(hsa_executable_iterate_agent_symbols) *hsa_executable_iterate_agent_symbols_fn;
  decltype(hsa_executable_iterate_program_symbols) *hsa_executable_iterate_program_symbols_fn;
};

/// @brief Top-level ROCR API table passed to HSA tools from `OnLoad()`.
///
/// @details The DBT MVP only patches `core_`. Extension tables are intentionally
/// opaque because queue interception, AMD extension APIs, finalizer APIs, image
/// APIs, tools events, and PC sampling are not part of the load-time DBT hook.
struct HsaApiTable {
  ApiTableVersion version;
  CoreApiTable *core_;
  AmdExtTable *amd_ext_;
  void *finalizer_ext_;
  void *image_ext_;
  void *tools_;
  void *pc_sampling_ext_;
};
