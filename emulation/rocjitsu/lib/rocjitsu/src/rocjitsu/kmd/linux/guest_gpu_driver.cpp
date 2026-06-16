// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/guest_gpu_driver.h"

#include "util/log.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <linux/memfd.h>
#include <optional>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

namespace fs = std::filesystem;

constexpr const char *kRealTopologyPath = "/sys/devices/virtual/kfd/kfd/topology";
constexpr const char *kKfdTopologyPrefix = "/sys/devices/virtual/kfd/kfd/topology";
constexpr const char *kKfdTopologyAltPrefix = "/sys/class/kfd/kfd/topology";
constexpr const char *kSysDevCharPrefix = "/sys/dev/char/";
constexpr uint64_t kKfdMmapTypeShift = 62;
constexpr uint64_t kKfdMmapTypeMask = 0x3ULL << kKfdMmapTypeShift;
constexpr uint64_t kKfdMmapTypeDoorbell = 0x3ULL << kKfdMmapTypeShift;
constexpr uint64_t kKfdMmapGpuIdShift = 46;

struct LinuxDirent64 {
  uint64_t d_ino;
  int64_t d_off;
  unsigned short d_reclen;
  unsigned char d_type;
  char d_name[1];
};

std::string read_text_file(const fs::path &path) {
  std::ifstream in(path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

void write_text_file(const fs::path &path, const std::string &text) {
  std::ofstream out(path);
  out << text;
}

uint32_t read_u32_property(const fs::path &path, std::string_view key, uint32_t fallback) {
  std::istringstream input(read_text_file(path));
  std::string name;
  uint64_t value = 0;
  while (input >> name >> value) {
    if (name == key)
      return static_cast<uint32_t>(value);
  }
  return fallback;
}

std::string set_property(std::string text, std::string_view key, uint64_t value) {
  std::istringstream input(text);
  std::ostringstream output;
  std::string line;
  bool replaced = false;
  while (std::getline(input, line)) {
    if (line.starts_with(key) && line.size() > key.size() && line[key.size()] == ' ') {
      output << key << " " << value << "\n";
      replaced = true;
    } else {
      output << line << "\n";
    }
  }
  if (!replaced)
    output << key << " " << value << "\n";
  return output.str();
}

bool make_temp_dir(const char *prefix, std::string *out) {
  char tmpl[128];
  std::snprintf(tmpl, sizeof(tmpl), "/tmp/%s_XXXXXX", prefix);
  char *dir = mkdtemp(tmpl);
  if (!dir)
    return false;
  *out = dir;
  return true;
}

uint32_t count_numeric_dirs(const fs::path &dir) {
  uint32_t count = 0;
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_directory(ec))
      continue;
    auto name = entry.path().filename().string();
    uint32_t ignored = 0;
    auto [ptr, err] = std::from_chars(name.data(), name.data() + name.size(), ignored);
    if (err == std::errc{} && ptr == name.data() + name.size())
      ++count;
  }
  return count;
}

std::optional<uint32_t> read_u32_file(const fs::path &path) {
  std::ifstream in(path);
  uint32_t value = 0;
  if (!(in >> value))
    return std::nullopt;
  return value;
}

void append_unique_gpu_id(std::vector<uint32_t> *ids, uint32_t gpu_id) {
  if (std::find(ids->begin(), ids->end(), gpu_id) == ids->end())
    ids->push_back(gpu_id);
}

uint32_t max_numeric_dir(const fs::path &dir) {
  uint32_t max_id = 0;
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_directory(ec))
      continue;
    auto name = entry.path().filename().string();
    uint32_t id = 0;
    auto [ptr, err] = std::from_chars(name.data(), name.data() + name.size(), id);
    if (err == std::errc{} && ptr == name.data() + name.size())
      max_id = std::max(max_id, id);
  }
  return max_id;
}

std::optional<uint32_t> first_real_gpu_id() {
  fs::path nodes_dir = fs::path(kRealTopologyPath) / "nodes";
  const uint32_t max_node = max_numeric_dir(nodes_dir);
  for (uint32_t node_id = 1; node_id <= max_node; ++node_id) {
    std::optional<uint32_t> gpu_id = read_u32_file(nodes_dir / std::to_string(node_id) / "gpu_id");
    if (gpu_id && *gpu_id != 0)
      return gpu_id;
  }
  return std::nullopt;
}

std::string io_link_to_cpu(uint32_t node_from) {
  std::ostringstream link;
  link << "type 2\n"
       << "version_major 0\n"
       << "version_minor 0\n"
       << "node_from " << node_from << "\n"
       << "node_to 0\n"
       << "weight 20\n"
       << "min_latency 0\n"
       << "max_latency 0\n"
       << "min_bandwidth 0\n"
       << "max_bandwidth 0\n"
       << "recommended_transfer_size 0\n"
       << "num_hops 1\n"
       << "flags 1\n";
  return link.str();
}

std::string io_link_from_cpu(uint32_t link_index, uint32_t node_to) {
  std::ostringstream link;
  (void)link_index;
  link << "type 2\n"
       << "version_major 0\n"
       << "version_minor 0\n"
       << "node_from 0\n"
       << "node_to " << node_to << "\n"
       << "weight 20\n"
       << "min_latency 0\n"
       << "max_latency 0\n"
       << "min_bandwidth 0\n"
       << "max_bandwidth 0\n"
       << "recommended_transfer_size 0\n"
       << "num_hops 1\n"
       << "flags 1\n";
  return link.str();
}

Sysfs::GpuInfo gpu_info_from_config(const config::KfdDeviceConfig &dev) {
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = dev.gpu_id;
  gpu.gfx_target_version = dev.gfx_target_version;
  gpu.vendor_id = dev.vendor_id;
  gpu.device_id = dev.device_id;
  gpu.family_id = dev.family_id;
  gpu.unique_id = dev.unique_id;
  gpu.location_id = dev.location_id;
  gpu.domain = dev.domain;
  gpu.hive_id = dev.hive_id;
  gpu.drm_render_minor = dev.drm_render_minor;
  gpu.marketing_name = dev.marketing_name.c_str();
  gpu.simd_count = dev.simd_count;
  gpu.max_waves_per_simd = dev.max_waves_per_simd;
  gpu.num_shader_engines = dev.num_shader_engines;
  gpu.num_shader_arrays_per_engine = dev.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = dev.num_cu_per_sh;
  gpu.simd_per_cu = dev.simd_per_cu;
  gpu.wave_front_size = dev.wave_front_size;
  gpu.max_slots_scratch_cu = dev.max_slots_scratch_cu;
  gpu.local_mem_size = dev.local_mem_size;
  gpu.lds_size_kb = dev.lds_size_kb;
  gpu.mem_width = dev.mem_width;
  gpu.mem_clk_max = dev.mem_clk_max;
  gpu.l1_size_kb = dev.l1_size_kb;
  gpu.l1_line_size = dev.l1_line_size;
  gpu.l1_assoc = dev.l1_assoc;
  gpu.l2_size_kb = dev.l2_size_kb;
  gpu.l2_line_size = dev.l2_line_size;
  gpu.l2_assoc = dev.l2_assoc;
  gpu.num_sdma_engines = dev.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = dev.num_sdma_xgmi_engines;
  gpu.num_cp_queues = dev.num_cp_queues;
  gpu.max_engine_clk_fcompute = dev.max_engine_clk_fcompute;
  gpu.num_xcc = std::max(1u, dev.num_shader_engines);
  return gpu;
}

uint32_t choose_render_minor(uint32_t requested) {
  auto available = [](uint32_t minor) {
    return !fs::exists("/sys/class/drm/renderD" + std::to_string(minor)) &&
           !fs::exists("/dev/dri/renderD" + std::to_string(minor));
  };

  if (requested >= 128 && available(requested))
    return requested;
  for (uint32_t minor = 224; minor < 512; ++minor) {
    if (available(minor))
      return minor;
  }
  return requested >= 128 ? requested : 255;
}

bool path_starts_with(std::string_view path, std::string_view prefix) {
  return path.starts_with(prefix);
}

bool raw_lstat(const std::string &path, struct stat *st) {
  return syscall(SYS_newfstatat, AT_FDCWD, path.c_str(), st, AT_SYMLINK_NOFOLLOW) == 0;
}

bool raw_copy_file(const std::string &src, const std::string &dst) {
  int in = static_cast<int>(syscall(SYS_openat, AT_FDCWD, src.c_str(), O_RDONLY | O_CLOEXEC));
  if (in < 0)
    return false;

  std::error_code ec;
  fs::create_directories(fs::path(dst).parent_path(), ec);
  if (ec) {
    syscall(SYS_close, in);
    return false;
  }

  int out = static_cast<int>(
      syscall(SYS_openat, AT_FDCWD, dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644));
  if (out < 0) {
    syscall(SYS_close, in);
    return false;
  }

  std::array<char, 4096> buffer{};
  for (;;) {
    ssize_t n = static_cast<ssize_t>(syscall(SYS_read, in, buffer.data(), buffer.size()));
    if (n == 0)
      break;
    if (n < 0) {
      syscall(SYS_close, out);
      syscall(SYS_close, in);
      return false;
    }
    char *p = buffer.data();
    ssize_t remaining = n;
    while (remaining > 0) {
      ssize_t written =
          static_cast<ssize_t>(syscall(SYS_write, out, p, static_cast<size_t>(remaining)));
      if (written <= 0) {
        syscall(SYS_close, out);
        syscall(SYS_close, in);
        return false;
      }
      p += written;
      remaining -= written;
    }
  }

  syscall(SYS_close, out);
  syscall(SYS_close, in);
  return true;
}

bool parse_render_minor(std::string_view path, uint32_t *minor, std::string_view *suffix) {
  constexpr std::string_view render_prefix = "/sys/class/drm/renderD";
  if (!path.starts_with(render_prefix))
    return false;
  auto rest = path.substr(render_prefix.size());
  auto slash = rest.find('/');
  auto number = slash == std::string_view::npos ? rest : rest.substr(0, slash);
  uint32_t parsed = 0;
  auto [ptr, err] = std::from_chars(number.data(), number.data() + number.size(), parsed);
  if (err != std::errc{} || ptr != number.data() + number.size())
    return false;
  *minor = parsed;
  *suffix = slash == std::string_view::npos ? std::string_view{} : rest.substr(slash);
  return true;
}

bool parse_sys_dev_render_minor(std::string_view path, uint32_t *minor, std::string_view *suffix) {
  if (!path.starts_with(kSysDevCharPrefix))
    return false;
  auto rest = path.substr(std::strlen(kSysDevCharPrefix));
  auto colon = rest.find(':');
  if (colon == std::string_view::npos)
    return false;
  uint32_t major = 0;
  auto [major_ptr, major_err] = std::from_chars(rest.data(), rest.data() + colon, major);
  if (major_err != std::errc{} || major_ptr != rest.data() + colon || major != 226)
    return false;

  auto after_colon = rest.substr(colon + 1);
  auto slash = after_colon.find('/');
  auto number = slash == std::string_view::npos ? after_colon : after_colon.substr(0, slash);
  uint32_t parsed = 0;
  auto [ptr, err] = std::from_chars(number.data(), number.data() + number.size(), parsed);
  if (err != std::errc{} || ptr != number.data() + number.size())
    return false;
  *minor = parsed;
  *suffix = slash == std::string_view::npos ? std::string_view{} : after_colon.substr(slash);
  return true;
}

const char *ioctl_name(unsigned long request) {
  switch (request) {
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return "ALLOC_MEMORY_OF_GPU";
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return "MAP_MEMORY_TO_GPU";
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return "UNMAP_MEMORY_FROM_GPU";
  case AMDKFD_IOC_CREATE_QUEUE:
    return "CREATE_QUEUE";
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
    return "SET_SCRATCH_BACKING_VA";
  case AMDKFD_IOC_SET_TRAP_HANDLER:
    return "SET_TRAP_HANDLER";
  case AMDKFD_IOC_SVM:
    return "SVM";
  default:
    return "guest-targeted execution ioctl";
  }
}

} // namespace

TopologyOverlay::~TopologyOverlay() { cleanup(); }

bool TopologyOverlay::copy_tree(const std::string &src, const std::string &dst) {
  std::error_code ec;
  fs::create_directories(dst, ec);
  if (ec)
    return false;

  int dir = static_cast<int>(
      syscall(SYS_openat, AT_FDCWD, src.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (dir < 0)
    return false;

  std::array<char, 32768> entries{};
  for (;;) {
    int nread = static_cast<int>(syscall(SYS_getdents64, dir, entries.data(), entries.size()));
    if (nread == 0)
      break;
    if (nread < 0) {
      syscall(SYS_close, dir);
      return false;
    }

    for (int pos = 0; pos < nread;) {
      auto *entry = reinterpret_cast<LinuxDirent64 *>(entries.data() + pos);
      std::string name = entry->d_name;
      pos += entry->d_reclen;
      if (name == "." || name == "..")
        continue;

      std::string child_src = src + "/" + name;
      std::string child_dst = dst + "/" + name;
      struct stat st {};
      if (!raw_lstat(child_src, &st))
        continue;

      if (S_ISDIR(st.st_mode)) {
        if (!copy_tree(child_src, child_dst)) {
          syscall(SYS_close, dir);
          return false;
        }
        continue;
      }
      if (S_ISLNK(st.st_mode)) {
        std::array<char, 4096> link_target{};
        ssize_t n = static_cast<ssize_t>(syscall(SYS_readlinkat, AT_FDCWD, child_src.c_str(),
                                                 link_target.data(), link_target.size() - 1));
        if (n > 0) {
          link_target[static_cast<size_t>(n)] = '\0';
          fs::create_directories(fs::path(child_dst).parent_path(), ec);
          fs::create_symlink(link_target.data(), child_dst, ec);
          ec.clear();
        }
        continue;
      }

      (void)raw_copy_file(child_src, child_dst);
    }
  }
  syscall(SYS_close, dir);
  return true;
}

bool TopologyOverlay::copy_guest_node(const Sysfs::GpuInfo &guest) {
  if (guest_sysfs_.generate(guest).empty())
    return false;
  guest_drm_dir_ = guest_sysfs_.drm_path();

  fs::path nodes_dir = fs::path(topology_dir_) / "nodes";
  guest_node_id_ = max_numeric_dir(nodes_dir) + 1;
  return copy_tree(fs::path(guest_sysfs_.path()) / "nodes" / "1",
                   nodes_dir / std::to_string(guest_node_id_));
}

bool TopologyOverlay::patch_topology_files() {
  fs::path root = topology_dir_;
  fs::path nodes_dir = root / "nodes";
  fs::path cpu_props = nodes_dir / "0" / "properties";
  fs::path system_props = root / "system_properties";
  fs::path guest_props = nodes_dir / std::to_string(guest_node_id_) / "properties";
  fs::path guest_link =
      nodes_dir / std::to_string(guest_node_id_) / "io_links" / "0" / "properties";

  uint32_t existing_links = read_u32_property(cpu_props, "io_links_count",
                                              count_numeric_dirs(nodes_dir / "0" / "io_links"));
  fs::path new_cpu_link = nodes_dir / "0" / "io_links" / std::to_string(existing_links);
  fs::create_directories(new_cpu_link);

  write_text_file(system_props, set_property(read_text_file(system_props), "num_devices",
                                             count_numeric_dirs(nodes_dir)));
  write_text_file(nodes_dir / "0" / "gpu_id", "0\n");
  write_text_file(cpu_props,
                  set_property(read_text_file(cpu_props), "io_links_count", existing_links + 1));
  write_text_file(new_cpu_link / "properties", io_link_from_cpu(existing_links, guest_node_id_));
  write_text_file(guest_props, set_property(read_text_file(guest_props), "io_links_count", 1));
  write_text_file(guest_link, io_link_to_cpu(guest_node_id_));
  return true;
}

bool TopologyOverlay::generate(const Sysfs::GpuInfo &guest) {
  cleanup();
  if (!make_temp_dir("rocjitsu_guest_topology", &topology_dir_))
    return false;
  if (!copy_tree(kRealTopologyPath, topology_dir_)) {
    cleanup();
    return false;
  }
  if (!copy_guest_node(guest) || !patch_topology_files()) {
    cleanup();
    return false;
  }
  return true;
}

void TopologyOverlay::cleanup() {
  if (!topology_dir_.empty()) {
    std::error_code ec;
    fs::remove_all(topology_dir_, ec);
    topology_dir_.clear();
  }
  guest_drm_dir_.clear();
  guest_node_id_ = 0;
  guest_sysfs_.cleanup();
}

GuestGPUDriver::GuestGPUDriver(config::DbtGuestConfig config) : config_(std::move(config)) {
  guest_ = gpu_info_from_config(config_.guest_device);
  guest_.drm_render_minor = choose_render_minor(guest_.drm_render_minor);
  host_gpu_id_ = config_.host_gpu_id;
}

GuestGPUDriver::~GuestGPUDriver() {
  if (synthetic_fd_ >= 0)
    syscall(SYS_close, synthetic_fd_);
  if (real_kfd_fd_ >= 0)
    syscall(SYS_close, real_kfd_fd_);
}

bool GuestGPUDriver::ensure_real_kfd() {
  if (real_kfd_fd_ >= 0)
    return true;
  int fd = static_cast<int>(syscall(SYS_openat, AT_FDCWD, "/dev/kfd", O_RDWR | O_CLOEXEC, 0));
  if (fd < 0)
    return false;
  real_kfd_fd_ = fd;
  return true;
}

bool GuestGPUDriver::ensure_ready() {
  std::lock_guard lock(mutex_);
  if (ready_)
    return true;
  if (!config_.enabled || !config_.guest_device.present) {
    errno = EINVAL;
    return false;
  }
  if (!ensure_real_kfd())
    return false;
  if (host_gpu_id_ == 0) {
    std::optional<uint32_t> first_host = first_real_gpu_id();
    if (!first_host) {
      errno = ENODEV;
      return false;
    }
    host_gpu_id_ = *first_host;
  }
  if (!overlay_.generate(guest_)) {
    errno = ENODEV;
    return false;
  }
  ready_ = true;
  return true;
}

int GuestGPUDriver::open() {
  if (!ensure_ready())
    return -1;
  if (synthetic_fd_ < 0) {
    synthetic_fd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_guest_kfd", MFD_CLOEXEC));
    if (synthetic_fd_ < 0)
      return -1;
  }
  return synthetic_fd_;
}

int GuestGPUDriver::close() { return 0; }

int GuestGPUDriver::forward_ioctl(unsigned long request, void *arg) {
  if (!ensure_real_kfd())
    return -1;
  return static_cast<int>(syscall(SYS_ioctl, real_kfd_fd_, request, arg));
}

int GuestGPUDriver::get_process_apertures_new(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }

  kfd_ioctl_get_process_apertures_new_args count_args{};
  if (forward_ioctl(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &count_args) != 0)
    return -1;
  const uint32_t host_count = count_args.num_of_nodes;
  const uint32_t total_count = host_count + 1;

  if (args->num_of_nodes == 0 || args->kfd_process_device_apertures_ptr == 0) {
    args->num_of_nodes = total_count;
    return 0;
  }

  auto *out = reinterpret_cast<kfd_process_device_apertures *>(
      static_cast<uintptr_t>(args->kfd_process_device_apertures_ptr));
  const uint32_t requested = args->num_of_nodes;
  std::vector<kfd_process_device_apertures> host(host_count);
  if (!host.empty()) {
    kfd_ioctl_get_process_apertures_new_args host_args{};
    host_args.kfd_process_device_apertures_ptr =
        reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(host.data()));
    host_args.num_of_nodes = host_count;
    if (forward_ioctl(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &host_args) != 0)
      return -1;
  }

  const uint32_t host_to_copy = std::min(requested, host_count);
  for (uint32_t i = 0; i < host_to_copy; ++i)
    out[i] = host[i];
  if (requested > host_count)
    out[host_count] = guest_apertures();
  args->num_of_nodes = std::min(requested, total_count);
  return 0;
}

int GuestGPUDriver::get_clock_counters(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_clock_counters_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id != guest_.gpu_id)
    return forward_ioctl(AMDKFD_IOC_GET_CLOCK_COUNTERS, arg);

  auto now = std::chrono::steady_clock::now().time_since_epoch();
  uint64_t ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  args->gpu_clock_counter = ns;
  args->cpu_clock_counter = ns;
  args->system_clock_counter = ns;
  args->system_clock_freq = 1000000000ULL;
  return 0;
}

int GuestGPUDriver::acquire_vm(void *arg) {
  auto *args = static_cast<kfd_ioctl_acquire_vm_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id == guest_.gpu_id)
    return 0;
  return forward_ioctl(AMDKFD_IOC_ACQUIRE_VM, arg);
}

int GuestGPUDriver::get_available_memory(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_available_memory_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id == guest_.gpu_id) {
    args->available = guest_.local_mem_size;
    return 0;
  }
  return forward_ioctl(AMDKFD_IOC_AVAILABLE_MEMORY, arg);
}

int GuestGPUDriver::set_memory_policy(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_memory_policy_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id == guest_.gpu_id)
    return 0;
  return forward_ioctl(AMDKFD_IOC_SET_MEMORY_POLICY, arg);
}

int GuestGPUDriver::alloc_memory(void *arg) {
  auto *args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id != guest_.gpu_id)
    return forward_ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, arg);

  std::lock_guard lock(mutex_);
  args->handle = next_synthetic_handle_++;
  synthetic_handles_.insert(args->handle);
  return 0;
}

int GuestGPUDriver::free_memory(void *arg) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  {
    std::lock_guard lock(mutex_);
    if (synthetic_handles_.erase(args->handle) != 0)
      return 0;
  }
  return forward_ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, arg);
}

int GuestGPUDriver::map_memory(void *arg) {
  auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }

  {
    std::lock_guard lock(mutex_);
    if (synthetic_handles_.count(args->handle) != 0) {
      args->n_success = args->n_devices;
      return 0;
    }
  }

  auto *ids =
      reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args->device_ids_array_ptr));
  if (!ids)
    return forward_ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, arg);

  std::vector<uint32_t> host_ids;
  host_ids.reserve(args->n_devices);
  bool has_guest = false;
  for (uint32_t i = 0; i < args->n_devices; ++i) {
    if (ids[i] == guest_.gpu_id) {
      has_guest = true;
      append_unique_gpu_id(&host_ids, host_gpu_id_);
    } else {
      append_unique_gpu_id(&host_ids, ids[i]);
    }
  }
  if (!has_guest)
    return forward_ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, arg);
  if (host_ids.empty()) {
    args->n_success = args->n_devices;
    return 0;
  }

  auto host_args = *args;
  host_args.device_ids_array_ptr =
      reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_ids.data()));
  host_args.n_devices = static_cast<uint32_t>(host_ids.size());
  host_args.n_success = 0;
  if (forward_ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &host_args) != 0)
    return -1;
  args->n_success = args->n_devices;
  return 0;
}

int GuestGPUDriver::unmap_memory(void *arg) {
  auto *args = static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }

  {
    std::lock_guard lock(mutex_);
    if (synthetic_handles_.count(args->handle) != 0) {
      args->n_success = args->n_devices;
      return 0;
    }
  }

  auto *ids =
      reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args->device_ids_array_ptr));
  if (!ids)
    return forward_ioctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, arg);

  std::vector<uint32_t> host_ids;
  host_ids.reserve(args->n_devices);
  bool has_guest = false;
  for (uint32_t i = 0; i < args->n_devices; ++i) {
    if (ids[i] == guest_.gpu_id) {
      has_guest = true;
      append_unique_gpu_id(&host_ids, host_gpu_id_);
    } else {
      append_unique_gpu_id(&host_ids, ids[i]);
    }
  }
  if (!has_guest)
    return forward_ioctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, arg);
  if (host_ids.empty()) {
    args->n_success = args->n_devices;
    return 0;
  }

  auto host_args = *args;
  host_args.device_ids_array_ptr =
      reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_ids.data()));
  host_args.n_devices = static_cast<uint32_t>(host_ids.size());
  host_args.n_success = 0;
  if (forward_ioctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &host_args) != 0)
    return -1;
  args->n_success = args->n_devices;
  return 0;
}

int GuestGPUDriver::ioctl(unsigned long request, void *arg) {
  if (!ensure_ready())
    return -1;

  switch (request) {
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    return get_process_apertures_new(arg);
  case AMDKFD_IOC_GET_CLOCK_COUNTERS:
    return get_clock_counters(arg);
  case AMDKFD_IOC_ACQUIRE_VM:
    return acquire_vm(arg);
  case AMDKFD_IOC_AVAILABLE_MEMORY:
    return get_available_memory(arg);
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return set_memory_policy(arg);
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return alloc_memory(arg);
  case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
    return free_memory(arg);
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return map_memory(arg);
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return unmap_memory(arg);
  case AMDKFD_IOC_GET_VERSION:
  case AMDKFD_IOC_SET_XNACK_MODE:
  case AMDKFD_IOC_RUNTIME_ENABLE:
    return forward_ioctl(request, arg);
  default:
    if (request_targets_guest(request, arg))
      return reject_guest_execution_ioctl(request, arg);
    return forward_ioctl(request, arg);
  }
}

void *GuestGPUDriver::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  if (!ensure_ready())
    return MAP_FAILED;
  uint64_t encoded_gpu = (static_cast<uint64_t>(offset) & ~kKfdMmapTypeMask) >> kKfdMmapGpuIdShift;
  if ((static_cast<uint64_t>(offset) & kKfdMmapTypeMask) == kKfdMmapTypeDoorbell &&
      encoded_gpu == guest_.gpu_id) {
    errno = ENODEV;
    return MAP_FAILED;
  }
  void *mapped =
      reinterpret_cast<void *>(syscall(SYS_mmap, addr, length, prot, flags, real_kfd_fd_, offset));
  if (reinterpret_cast<intptr_t>(mapped) < 0)
    return MAP_FAILED;
  return mapped;
}

int GuestGPUDriver::munmap(void *, size_t) { return -ENOENT; }

bool GuestGPUDriver::owns_fd(int fd) const { return fd >= 0 && fd == synthetic_fd_; }

std::string GuestGPUDriver::redirect_sysfs_path(const char *path) const {
  if (!path || overlay_.topology_path().empty())
    return {};
  std::string_view sv(path);
  if (path_starts_with(sv, kKfdTopologyPrefix))
    return overlay_.topology_path() + std::string(sv.substr(std::strlen(kKfdTopologyPrefix)));
  if (path_starts_with(sv, kKfdTopologyAltPrefix))
    return overlay_.topology_path() + std::string(sv.substr(std::strlen(kKfdTopologyAltPrefix)));

  uint32_t minor = 0;
  std::string_view suffix;
  if (parse_render_minor(sv, &minor, &suffix) && minor == guest_.drm_render_minor)
    return overlay_.guest_drm_path() + "/renderD" + std::to_string(minor) + std::string(suffix);
  if (parse_sys_dev_render_minor(sv, &minor, &suffix) && minor == guest_.drm_render_minor)
    return overlay_.guest_drm_path() + "/renderD" + std::to_string(minor) + std::string(suffix);

  return {};
}

bool GuestGPUDriver::is_doorbell_range(const void *, size_t) const { return false; }

bool GuestGPUDriver::handles_drm_render_minor(uint32_t minor) const {
  return ready_ && minor == guest_.drm_render_minor;
}

const Sysfs::GpuInfo *GuestGPUDriver::gpu_info_for_render_minor(uint32_t minor) const {
  return handles_drm_render_minor(minor) ? &guest_ : nullptr;
}

int GuestGPUDriver::reject_guest_execution_ioctl(unsigned long request, void *) const {
  util::Logger::debug_print("rocjitsu guest gpu: rejecting ", ioctl_name(request),
                            " for guest gpu_id=", guest_.gpu_id);
  errno = ENODEV;
  return -1;
}

bool GuestGPUDriver::request_targets_guest(unsigned long request, void *arg) const {
  if (!arg)
    return false;
  switch (request) {
  case AMDKFD_IOC_CREATE_QUEUE:
    return static_cast<kfd_ioctl_create_queue_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return static_cast<kfd_ioctl_set_memory_policy_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_IMPORT_DMABUF:
    return static_cast<kfd_ioctl_import_dmabuf_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_SMI_EVENTS:
    return static_cast<kfd_ioctl_smi_events_args *>(arg)->gpuid == guest_.gpu_id;
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
    return static_cast<kfd_ioctl_set_scratch_backing_va_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_GET_TILE_CONFIG:
    return static_cast<kfd_ioctl_get_tile_config_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_SET_TRAP_HANDLER:
    return static_cast<kfd_ioctl_set_trap_handler_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU: {
    auto *a = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);
    auto *ids = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(a->device_ids_array_ptr));
    return ids && std::find(ids, ids + a->n_devices, guest_.gpu_id) != ids + a->n_devices;
  }
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
    auto *a = static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arg);
    auto *ids = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(a->device_ids_array_ptr));
    return ids && std::find(ids, ids + a->n_devices, guest_.gpu_id) != ids + a->n_devices;
  }
  case AMDKFD_IOC_SVM: {
    auto *a = static_cast<kfd_ioctl_svm_args *>(arg);
    for (uint32_t i = 0; i < a->nattr; ++i) {
      auto &attr = a->attrs[i];
      if ((attr.type == KFD_IOCTL_SVM_ATTR_PREFERRED_LOC ||
           attr.type == KFD_IOCTL_SVM_ATTR_PREFETCH_LOC || attr.type == KFD_IOCTL_SVM_ATTR_ACCESS ||
           attr.type == KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE ||
           attr.type == KFD_IOCTL_SVM_ATTR_NO_ACCESS) &&
          attr.value == guest_.gpu_id)
        return true;
    }
    return false;
  }
  default:
    return false;
  }
}

kfd_process_device_apertures GuestGPUDriver::guest_apertures() const {
  const uint64_t ordinal = std::max<uint32_t>(1, overlay_.guest_node_id());
  const uint64_t aperture_stride = 0x10000000000ULL;
  kfd_process_device_apertures apertures{};
  apertures.lds_base = 0x1000000000000ULL + ordinal * aperture_stride;
  apertures.lds_limit = apertures.lds_base + 0xFFFFFFFFULL;
  apertures.scratch_base = 0x2000000000000ULL + ordinal * aperture_stride;
  apertures.scratch_limit = apertures.scratch_base + 0xFFFFFFFFULL;
  apertures.gpuvm_base = 0x1000000000ULL;
  apertures.gpuvm_limit = 0x3FFFFFFFFFFFULL;
  apertures.gpu_id = guest_.gpu_id;
  return apertures;
}

} // namespace rocjitsu
