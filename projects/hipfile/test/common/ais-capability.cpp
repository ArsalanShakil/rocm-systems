/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "ais-capability.h"

#include "hip.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/utsname.h>
#include <zlib.h>

namespace hipFile::test {

namespace {

    bool kernelSupportsP2pdma()
    {
        struct utsname uts;
        if (::uname(&uts) != 0) {
            std::cerr << "uname() failed; cannot determine kernel release\n";
            return false;
        }
        const std::string release{uts.release};

        const std::string plain_config = "/boot/config-" + release;
        const std::string build_config = "/lib/modules/" + release + "/build/.config";

        auto is_match = [](const std::string &line) {
            return line.rfind("CONFIG_PCI_P2PDMA=y", 0) == 0 || line.rfind("CONFIG_PCI_P2PDMA=m", 0) == 0;
        };

        bool configs_found = false;

        // Plain-text config sources.
        for (const std::string &path : {plain_config, build_config}) {
            std::ifstream in{path};
            if (!in.is_open()) {
                continue;
            }
            configs_found = true;
            std::string line;
            while (std::getline(in, line)) {
                if (is_match(line)) {
                    return true;
                }
            }
        }

        // Gzipped config exposed by the running kernel.
        if (gzFile gz = gzopen("/proc/config.gz", "rb")) {
            configs_found = true;
            char buf[512];
            while (gzgets(gz, buf, static_cast<int>(sizeof(buf))) != nullptr) {
                if (is_match(std::string{buf})) {
                    gzclose(gz);
                    return true;
                }
            }
            gzclose(gz);
        }

        if (!configs_found) {
            std::cerr << "No kernel config files found!\n";
        }
        return false;
    }

    bool hipRuntimeSupportsAis()
    {
        return hipFile::getHipAmdFileReadPtr() != nullptr && hipFile::getHipAmdFileWritePtr() != nullptr;
    }

    bool amdgpuSupportsAis()
    {
        std::ifstream kallsyms{"/proc/kallsyms"};
        if (!kallsyms.is_open()) {
            std::cerr << "Unable to open /proc/kallsyms\n";
            return false;
        }

        std::string line;
        while (std::getline(kallsyms, line)) {
            if (line.find("kfd_ais_rw_file") != std::string::npos) {
                return true;
            }
        }
        return false;
    }

}

// Reimplements logic from hipfile/tools/ais-check/ais-check.
AisCapability
detectAisCapability()
{
    AisCapability cap;
    cap.kernel_p2pdma = kernelSupportsP2pdma();
    cap.hip_runtime   = hipRuntimeSupportsAis();
    cap.amdgpu        = amdgpuSupportsAis();

    std::cerr << "AIS kernel P2PDMA support: " << (cap.kernel_p2pdma ? "yes" : "no") << "\n";
    std::cerr << "AIS HIP runtime support:   " << (cap.hip_runtime ? "yes" : "no") << "\n";
    std::cerr << "AIS amdgpu support:        " << (cap.amdgpu ? "yes" : "no") << "\n";

    return cap;
}

bool
fastpathAvailable()
{
    return detectAisCapability().fastpath_available();
}

}
