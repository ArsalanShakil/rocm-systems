/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace hipFile::test {

// Check AIS capability for tests that attempt to force fast path.
// Reimplements logic from hipfile/tools/ais-check/ais-check.
struct AisCapability {
    bool kernel_p2pdma; ///< CONFIG_PCI_P2PDMA=y|m present in a kernel config
    bool hip_runtime;   ///< hipAmdFileRead + hipAmdFileWrite resolvable
    bool amdgpu;        ///< kfd_ais_rw_file present in /proc/kallsyms

    bool fastpath_available() const
    {
        return kernel_p2pdma && hip_runtime && amdgpu;
    }
};

AisCapability detectAisCapability();

bool fastpathAvailable();

}
