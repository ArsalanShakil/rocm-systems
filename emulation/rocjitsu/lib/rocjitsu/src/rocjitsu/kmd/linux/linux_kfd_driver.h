// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file linux_kfd_driver.h
/// @brief Linux-specific KFD driver interface used by the syscall interposer.

#ifndef ROCJITSU_KMD_LINUX_LINUX_KFD_DRIVER_H_
#define ROCJITSU_KMD_LINUX_LINUX_KFD_DRIVER_H_

#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/driver.h"

#include <cstdint>
#include <string>

namespace rocjitsu {

/// @brief Linux KFD driver surface required by the LD_PRELOAD shim.
///
/// @details The base Driver interface handles /dev/kfd calls. The Linux shim
/// also needs sysfs redirection, DRM render-node ownership, and fd tracking.
/// SimulatedDriver and GuestGPUDriver both implement this surface, but with
/// different ownership rules: simulation owns all visible GPU discovery, while
/// GuestGPUDriver owns only the appended guest GPU.
class LinuxKfdDriver : public Driver {
public:
  ~LinuxKfdDriver() override = default;

  /// @brief Return the synthetic /dev/kfd fd owned by this driver.
  [[nodiscard]] virtual int fd() const = 0;

  /// @brief Return true when @p fd is owned internally by the driver.
  [[nodiscard]] virtual bool owns_fd(int fd) const = 0;

  /// @brief Redirect a sysfs path to a generated topology/DRM path.
  ///
  /// @details Returns an empty string when the driver does not own the path.
  [[nodiscard]] virtual std::string redirect_sysfs_path(const char *path) const = 0;

  /// @brief Return true if a memory range is a KFD doorbell mapping.
  [[nodiscard]] virtual bool is_doorbell_range(const void *addr, size_t length) const = 0;

  /// @brief Return true if this driver should synthesize /dev/dri/renderD@p minor.
  [[nodiscard]] virtual bool handles_drm_render_minor(uint32_t minor) const = 0;

  /// @brief Return synthetic GPU properties for a handled DRM render minor.
  [[nodiscard]] virtual const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const = 0;

  /// @brief Return the generated KFD topology root, if any.
  [[nodiscard]] virtual std::string topology_path() const = 0;

  /// @brief Return a generated /sys/class/drm root for full DRM redirection.
  ///
  /// @details GuestGPUDriver returns an empty string because host DRM entries
  /// must remain real; it redirects only the synthetic guest render node.
  [[nodiscard]] virtual std::string drm_path() const = 0;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_LINUX_KFD_DRIVER_H_
