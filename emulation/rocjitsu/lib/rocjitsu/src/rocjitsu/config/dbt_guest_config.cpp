// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/dbt_guest_config.h"

#include "embedded_schema.h"
#include "flatbuffers/idl.h"
#include "simulation_config_generated.h"

#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rocjitsu::config {
namespace {

std::string read_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Cannot open file: " + path);

  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

const fb::SimulationConfig *parse_json(const std::string &json, flatbuffers::Parser *parser) {
  parser->opts.skip_unexpected_fields_in_json = true;
  if (!parser->Parse(rocjitsu::kEmbeddedSchema))
    throw std::runtime_error("Failed to parse schema: " + std::string(parser->error_));
  if (!parser->Parse(json.c_str()))
    throw std::runtime_error("Failed to parse JSON config: " + std::string(parser->error_));
  return flatbuffers::GetRoot<fb::SimulationConfig>(parser->builder_.GetBufferPointer());
}

KfdDeviceConfig kfd_device_from_fb(const fb::KfdDeviceInfo *device) {
  KfdDeviceConfig config;
  if (device == nullptr)
    return config;

  config.present = true;
  config.gpu_id = device->gpu_id();
  config.gfx_target_version = device->gfx_target_version();
  config.vendor_id = device->vendor_id();
  config.device_id = device->device_id();
  config.family_id = device->family_id();
  config.unique_id = device->unique_id();
  if (device->marketing_name())
    config.marketing_name = device->marketing_name()->str();
  config.drm_render_minor = device->drm_render_minor();
  config.simd_count = device->simd_count();
  config.max_waves_per_simd = device->max_waves_per_simd();
  config.num_shader_engines = device->num_shader_engines();
  config.num_shader_arrays_per_engine = device->num_shader_arrays_per_engine();
  config.num_cu_per_sh = device->num_cu_per_sh();
  config.simd_per_cu = device->simd_per_cu();
  config.wave_front_size = device->wave_front_size();
  config.max_slots_scratch_cu = device->max_slots_scratch_cu();
  config.local_mem_size = device->local_mem_size();
  config.lds_size_kb = device->lds_size_kb();
  config.mem_width = device->mem_width();
  config.mem_clk_max = device->mem_clk_max();
  config.l1_size_kb = device->l1_size_kb();
  config.l1_line_size = device->l1_line_size();
  config.l1_assoc = device->l1_assoc();
  config.l2_size_kb = device->l2_size_kb();
  config.l2_line_size = device->l2_line_size();
  config.l2_assoc = device->l2_assoc();
  config.num_sdma_engines = device->num_sdma_engines();
  config.num_sdma_xgmi_engines = device->num_sdma_xgmi_engines();
  config.num_cp_queues = device->num_cp_queues();
  config.max_engine_clk_fcompute = device->max_engine_clk_fcompute();
  config.location_id = device->location_id();
  config.hive_id = device->hive_id();
  config.domain = device->domain();
  return config;
}

DbtGuestConfig dbt_guest_from_fb(const fb::DbtGuestConfig *guest) {
  DbtGuestConfig config;
  if (guest == nullptr)
    return config;

  config.enabled = guest->enabled();
  if (guest->guest_isa())
    config.guest_isa = guest->guest_isa()->str();
  if (guest->host_isa())
    config.host_isa = guest->host_isa()->str();
  config.host_gpu_id = guest->host_gpu_id();
  config.log_level = guest->log_level();
  config.signal_backtrace = guest->signal_backtrace();
  config.guest_device = kfd_device_from_fb(guest->guest_device());
  return config;
}

} // namespace

std::optional<DbtGuestConfig> load_dbt_guest_config_from_rj_config() {
  const char *path = std::getenv("RJ_CONFIG");
  if (path == nullptr || *path == '\0')
    return std::nullopt;

  flatbuffers::Parser parser;
  const fb::SimulationConfig *config = parse_json(read_file(path), &parser);
  return dbt_guest_from_fb(config->dbt_guest());
}

} // namespace rocjitsu::config
