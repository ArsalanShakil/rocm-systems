/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// =============================================================================
// ROCR Runtime Logging System
// =============================================================================
//
// Unified logging with CLR/HIP. Disabled by default.
//
// CLR/HIP INTEGRATION:
//   When AMD_LOG_LEVEL >= 6, CLR enables ROCR logging via hsa_amd_enable_logging().
//   Both CLR and ROCR logs go to the same file for unified debugging.
//
//   For HIP applications, use AMD_LOG_LEVEL for unified CLR+ROCR+Thunk logging:
//     AMD_LOG_LEVEL=6 ./hip_app
//
// STANDALONE HSA APPLICATIONS:
//   HSA_LOG_LEVEL=help ./app    Show usage help
//   HSA_LOG_LEVEL=3 ./app       Enable info-level logging
//
// =============================================================================

#ifndef HSA_RUNTIME_CORE_UTIL_LOGGING_H_
#define HSA_RUNTIME_CORE_UTIL_LOGGING_H_

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <io.h>
#define STDERR_FILENO 2
#else
#include <unistd.h>
#endif

namespace rocr {

// ============================================================================
// Log Levels - Severity levels for log messages
// ============================================================================
enum LogLevel {
  LOG_NONE = 0,     // No logging
  LOG_ERROR = 1,    // Errors only
  LOG_WARNING = 2,  // Warnings + errors
  LOG_INFO = 3,     // General info
  LOG_DEBUG = 4,    // Detailed debug
  LOG_TRACE = 5,    // Very detailed tracing
  LOG_VERBOSE = 6   // Extremely detailed
};

// ============================================================================
// Log Categories - Simplified per review feedback (MEM/INFO/ERROR)
// ============================================================================
enum LogMask : uint64_t {
  // Primary categories (per reviewer feedback)
  LOG_MEM = 0x1,        // Memory allocations/free
  LOG_INFO_CAT = 0x2,   // General info logs
  LOG_ERROR_CAT = 0x4,  // Error logs

  // Additional categories mapped to above for compatibility
  LOG_INIT = LOG_INFO_CAT,     // Runtime init/shutdown -> INFO
  LOG_QUEUE = LOG_INFO_CAT,    // Queue operations -> INFO
  LOG_SIGNAL = LOG_INFO_CAT,   // Signal operations -> INFO
  LOG_IPC = LOG_MEM,           // IPC -> MEM (memory sharing)
  LOG_AGENT = LOG_INFO_CAT,    // Agent/topology -> INFO
  LOG_AQL = LOG_INFO_CAT,      // AQL packets -> INFO
  LOG_SDMA = LOG_INFO_CAT,     // SDMA operations -> INFO
  LOG_COPY = LOG_MEM,          // Copy operations -> MEM
  LOG_BLIT = LOG_INFO_CAT,     // BlitKernel -> INFO
  LOG_SCRATCH = LOG_MEM,       // Scratch allocation -> MEM
  LOG_POOL = LOG_MEM,          // Memory/signal pools -> MEM
  LOG_EXCEPT = LOG_ERROR_CAT,  // Exception handlers -> ERROR

  // Always log (matches all masks)
  LOG_ALWAYS = 0xFFFFFFFFFFFFFFFFULL
};

// Legacy compatibility aliases for categories
constexpr uint64_t ROCR_LOG_MEM = LOG_MEM;
constexpr uint64_t ROCR_LOG_INIT = LOG_INIT;
constexpr uint64_t ROCR_LOG_QUEUE = LOG_QUEUE;
constexpr uint64_t ROCR_LOG_SIGNAL = LOG_SIGNAL;
constexpr uint64_t ROCR_LOG_IPC = LOG_IPC;
constexpr uint64_t ROCR_LOG_AGENT = LOG_AGENT;
constexpr uint64_t ROCR_LOG_AQL = LOG_AQL;
constexpr uint64_t ROCR_LOG_SDMA = LOG_SDMA;
constexpr uint64_t ROCR_LOG_COPY = LOG_COPY;
constexpr uint64_t ROCR_LOG_BLIT = LOG_BLIT;
constexpr uint64_t ROCR_LOG_SCRATCH = LOG_SCRATCH;
constexpr uint64_t ROCR_LOG_POOL = LOG_POOL;
constexpr uint64_t ROCR_LOG_EXCEPT = LOG_EXCEPT;
// Additional legacy aliases (mapped to simplified categories)
constexpr uint64_t ROCR_LOG_FAULT = LOG_ERROR_CAT;
constexpr uint64_t ROCR_LOG_PERF = LOG_INFO_CAT;
constexpr uint64_t ROCR_LOG_HEALTH = LOG_INFO_CAT;
constexpr uint64_t ROCR_LOG_TOPOLOGY = LOG_INFO_CAT;
constexpr uint64_t ROCR_LOG_API = LOG_INFO_CAT;
constexpr uint64_t ROCR_LOG_WAIT = LOG_INFO_CAT;
constexpr uint64_t ROCR_LOG_HANG = LOG_ERROR_CAT;
constexpr uint64_t ROCR_LOG_ALWAYS = LOG_ALWAYS;

// Legacy level aliases
constexpr int ROCR_LOG_NONE = LOG_NONE;
constexpr int ROCR_LOG_ERROR = LOG_ERROR;
constexpr int ROCR_LOG_WARNING = LOG_WARNING;
constexpr int ROCR_LOG_INFO = LOG_INFO;
constexpr int ROCR_LOG_DEBUG = LOG_DEBUG;
constexpr int ROCR_LOG_TRACE = LOG_TRACE;
constexpr int ROCR_LOG_VERBOSE = LOG_VERBOSE;

// ============================================================================
// Global Logging State
// ============================================================================
struct LoggingState {
  int log_level;          // Current log level (from HSA_LOG_LEVEL)
  uint64_t log_mask;      // Current log mask (from HSA_LOG_MASK)
  FILE* log_file;         // Log file handle (or stderr)
  bool owns_log_file;     // True if we opened log_file and should close it
  bool initialized;       // Whether logging is initialized
  bool clr_controlled;    // True if AMD_LOG_LEVEL is set (CLR controls logging)
  std::mutex file_mutex;  // Mutex for file operations

  LoggingState()
      : log_level(0),
        log_mask(0xFFFFFFFFFFFFFFFFULL),
        log_file(stderr),
        owns_log_file(false),
        initialized(false),
        clr_controlled(false) {}
};

// Global logging state instance
extern LoggingState g_log_state;

// Legacy alias for compatibility
#define g_rocr_log_state g_log_state

// ============================================================================
// Core Logging Functions
// ============================================================================

// Initialize the logging system (called from Runtime::Load)
void log_init();

// Shutdown the logging system (called from Runtime::Unload)
void log_shutdown();

// Core logging function
void log_printf(int level, uint64_t mask, const char* file, int line, const char* format, ...);

// Get current timestamp in microseconds
uint64_t get_timestamp_us();

// Legacy aliases
inline void rocr_log_init() { log_init(); }
inline void rocr_log_shutdown() { log_shutdown(); }
inline uint64_t rocr_get_timestamp_us() { return get_timestamp_us(); }

// ============================================================================
// Logging Macros
// ============================================================================

// Check if logging is enabled for a given level and mask
#define LOG_ENABLED(level, mask)                                                                   \
  (rocr::g_log_state.log_level >= (level) && (rocr::g_log_state.log_mask & (mask)))

// Legacy alias
#define ROCR_LOG_ENABLED(level, mask) LOG_ENABLED(level, mask)

// Get just the filename from full path
#ifdef _WIN32
#define __LOG_FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __LOG_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

// Legacy alias
#define __ROCR_FILENAME__ __LOG_FILENAME__

// Main logging macro
#define Log(level, mask, format, ...)                                                              \
  do {                                                                                             \
    if (LOG_ENABLED(level, mask)) {                                                                \
      rocr::log_printf(level, mask, __LOG_FILENAME__, __LINE__, format, ##__VA_ARGS__);            \
    }                                                                                              \
  } while (false)

// Convenience macros for different log levels
#define LogError(mask, format, ...) Log(rocr::LOG_ERROR, mask, format, ##__VA_ARGS__)

#define LogWarning(mask, format, ...) Log(rocr::LOG_WARNING, mask, format, ##__VA_ARGS__)

#define LogInfo(mask, format, ...) Log(rocr::LOG_INFO, mask, format, ##__VA_ARGS__)

#define LogDebug(mask, format, ...) Log(rocr::LOG_DEBUG, mask, format, ##__VA_ARGS__)

// Legacy aliases with Rocr prefix
#define RocrLog Log
#define RocrLogError LogError
#define RocrLogWarning LogWarning
#define RocrLogInfo LogInfo
#define RocrLogDebug LogDebug
#define RocrLogTrace(mask, format, ...) Log(rocr::LOG_TRACE, mask, format, ##__VA_ARGS__)
#define RocrLogVerbose(mask, format, ...) Log(rocr::LOG_VERBOSE, mask, format, ##__VA_ARGS__)

// ============================================================================
// Function Tracing - Entry/Exit logging
// ============================================================================

#define ROCR_TRACE_ENTER(mask, format, ...)                                                        \
  do {                                                                                             \
    if (LOG_ENABLED(rocr::LOG_VERBOSE, mask)) {                                                    \
      rocr::log_printf(rocr::LOG_VERBOSE, mask, __LOG_FILENAME__, __LINE__,                        \
                       "ENTER %s(" format ")", __func__, ##__VA_ARGS__);                           \
    }                                                                                              \
  } while (false)

#define ROCR_TRACE_EXIT(mask, format, ...)                                                         \
  do {                                                                                             \
    if (LOG_ENABLED(rocr::LOG_VERBOSE, mask)) {                                                    \
      rocr::log_printf(rocr::LOG_VERBOSE, mask, __LOG_FILENAME__, __LINE__, "EXIT %s" format,      \
                       __func__, ##__VA_ARGS__);                                                   \
    }                                                                                              \
  } while (false)

#define ROCR_TRACE_EXIT_STATUS(mask, status) ROCR_TRACE_EXIT(mask, " -> 0x%x", (unsigned)(status))

// ============================================================================
// IPC Leak Detection Warning
// ============================================================================

// Warn on high IPC attachment counts (potential leak detection)
void log_ipc_warning(uint64_t active_count, uint64_t total_count, void* ptr);

}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_LOGGING_H_
