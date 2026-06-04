# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from typing import Optional

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs


class gfx1151_soc(OmniSoC_Base):
    # LPDDR5X memory: 256-bit total bus width / 32-bit per channel = 8 channels.
    # gfx1151 (Strix Halo) uses LPDDR5X, not HBM.
    _NUM_MEMORY_CHANNELS: int = 8

    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx1151")
        self.set_compatible_profilers(["rocprofv3", "rocprofiler-sdk"])
        # Per IP block max number of simultaneous counters. GFX IP Blocks
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx1151"))

        # Set arch specific specs for RDNA3.5
        self._mspec.l2_banks = 8
        self._mspec.lds_banks_per_cu = 32
        self._mspec.pipes_per_gpu = 2

        # LPDDR5X: 256-bit bus / 32-bit per channel = 8 memory channels.
        # This is set here so generate_machine_specs() skips the NPS-based
        # HBM channel derivation (which is MI-GPU-specific and not applicable
        # to this LPDDR5X APU).
        self._mspec.num_memory_channels = str(self._NUM_MEMORY_CHANNELS)

        # GL1 (Shader Array) cache count: RDNA3.5 has 4 CUs per Shader Array.
        # num_gl1c is used by analysis config formulas for GL1 bandwidth ceilings.
        if self._mspec.cu_per_gpu is not None:
            self._mspec.num_gl1c = str(int(self._mspec.cu_per_gpu) // 4)
        else:
            self._mspec.num_gl1c = None

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        # Performance counter filtering
        filter_blocks = self.perfmon_filter()
        return filter_blocks

    @demarcate
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()
