# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(FetchContent)
FetchContent_Declare(
  libassert
  GIT_REPOSITORY https://github.com/jeremy-rifkin/libassert.git
  GIT_TAG        bd33ba116f209bf71761c58dccc2f3bf277e0824 # v2.2.1
)
FetchContent_MakeAvailable(libassert)

# libassert's ASSERT macro is variadic (expr, ...). Calling it with just an expression
# and no extra message args passes zero arguments for '...', which is a C++20 extension.
# Suppress for all consumers since this is inherent to the library's macro design.
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20.1)
    target_compile_options(libassert-lib INTERFACE -Wno-variadic-macro-arguments-omitted)
endif()
