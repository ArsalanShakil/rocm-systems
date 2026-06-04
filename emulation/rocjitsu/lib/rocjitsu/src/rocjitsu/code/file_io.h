// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_FILE_IO_H_
#define ROCJITSU_CODE_FILE_IO_H_

#include <string_view>
#include <vector>

namespace rocjitsu::detail {

std::vector<char> read_file_bytes(std::string_view path);

} // namespace rocjitsu::detail

#endif // ROCJITSU_CODE_FILE_IO_H_
