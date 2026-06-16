// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_guest_config.h
/// @brief DBT guest-GPU configuration shared by the launcher, KMD interposer, and HSA hook.

#ifndef ROCJITSU_CONFIG_DBT_GUEST_CONFIG_H_
#define ROCJITSU_CONFIG_DBT_GUEST_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

namespace rocjitsu::config {

/// @brief KFD device identity extracted from vm.gpu.device or dbt_guest.guest_device.
///
/// @details The KMD interposer copies these values into generated KFD topology
/// files and synthetic DRM/AMDGPU metadata. In DBT guest mode the values
/// describe the guest GPU that applications see, not the host GPU that executes
/// translated code.
struct KfdDeviceConfig {
  uint32_t gpu_id = 0;              ///< KFD gpu_id advertised in topology and ioctls.
  uint32_t gfx_target_version = 0;  ///< KFD gfx target version, for example 90500 for gfx950.
  uint32_t vendor_id = 0x1002;      ///< PCI vendor id used by synthetic DRM metadata.
  uint32_t device_id = 0;           ///< PCI device id used by synthetic DRM metadata.
  uint32_t family_id = 0;           ///< AMDGPU family id reported for the guest.
  uint64_t unique_id = 0;           ///< Stable synthetic unique id for this node.
  std::string marketing_name;       ///< User-visible GPU name, for example MI350X.
  uint32_t drm_render_minor = 128;  ///< Preferred synthetic /dev/dri/renderD minor.
  uint32_t simd_count = 0;          ///< Total SIMD units exposed in KFD properties.
  uint32_t max_waves_per_simd = 10; ///< Maximum waves per SIMD.
  uint32_t num_shader_engines = 0;  ///< Shader engine count.
  uint32_t num_shader_arrays_per_engine = 1; ///< Shader arrays per shader engine.
  uint32_t num_cu_per_sh = 0;                ///< Compute units per shader array.
  uint32_t simd_per_cu = 4;                  ///< SIMD units per compute unit.
  uint32_t wave_front_size = 64;             ///< Wavefront width reported to runtimes.
  uint32_t max_slots_scratch_cu = 32;        ///< Scratch slots per compute unit.
  uint64_t local_mem_size = 0;               ///< Guest-visible local memory size.
  uint32_t lds_size_kb = 64;                 ///< LDS size per CU in KiB.
  uint32_t mem_width = 4096;                 ///< Memory interface width in bits.
  uint32_t mem_clk_max = 1200;               ///< Maximum memory clock in MHz.
  uint32_t l1_size_kb = 32;                  ///< L1 cache size in KiB.
  uint32_t l1_line_size = 128;               ///< L1 cache line size in bytes.
  uint32_t l1_assoc = 4;                     ///< L1 cache associativity.
  uint32_t l2_size_kb = 4096;                ///< L2 cache size in KiB.
  uint32_t l2_line_size = 128;               ///< L2 cache line size in bytes.
  uint32_t l2_assoc = 16;                    ///< L2 cache associativity.
  uint32_t num_sdma_engines = 2;             ///< SDMA engine count.
  uint32_t num_sdma_xgmi_engines = 0;        ///< XGMI SDMA engine count.
  uint32_t num_cp_queues = 128;              ///< Hardware queue count.
  uint32_t max_engine_clk_fcompute = 2100;   ///< Maximum compute clock in MHz.
  uint32_t location_id = 0x0300;             ///< PCI BDF location id.
  uint64_t hive_id = 0;                      ///< XGMI hive id.
  uint32_t domain = 0;                       ///< PCI domain.
  bool present = false;                      ///< True if device section existed in config.
};

/// @brief DBT guest-GPU discovery configuration.
///
/// @details When enabled, the Linux KFD interposer exposes one synthetic guest
/// GPU alongside the real host GPUs and the HSA tools hook maps guest-agent
/// execution calls to the host agent. The KFD layer only owns discovery; DBT and
/// HSA forwarding happen in the HSA hook.
struct DbtGuestConfig {
  bool enabled = false;          ///< True when GuestGPUDriver mode is active.
  std::string guest_isa;         ///< Guest ISA advertised by the synthetic agent.
  std::string host_isa;          ///< Host ISA used for actual ROCR execution.
  uint32_t host_gpu_id = 0;      ///< Host KFD topology gpu_id; 0 selects first host_isa agent.
  int log_level = 0;             ///< DBT hook logging level loaded from the config file.
  bool signal_backtrace = false; ///< Install an HSA-hook crash backtrace handler.
  KfdDeviceConfig guest_device;  ///< Synthetic guest device appended to KFD topology.
};

/// @brief Load only dbt_guest from the rocjitsu child-process config named by RJ_CONFIG.
///
/// @details HSA tools run inside ROCR initialization and must not depend on the
/// full simulation topology builder. This helper parses the same JSON schema as
/// the main config loader but copies only the DBT guest-GPU block.
/// @returns DbtGuestConfig when RJ_CONFIG is set; std::nullopt when it is unset.
/// @throws std::runtime_error on file I/O, parse errors, or invalid config.
std::optional<DbtGuestConfig> load_dbt_guest_config_from_rj_config();

} // namespace rocjitsu::config

#endif // ROCJITSU_CONFIG_DBT_GUEST_CONFIG_H_
