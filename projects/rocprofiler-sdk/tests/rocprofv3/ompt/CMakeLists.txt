#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-ompt-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

set(PYTEST_ARGS)
if(ROCPROFILER_MEMCHECK MATCHES "(Address|Thread)Sanitizer" OR ROCPROFILER_BUILD_CODECOV)
    set(PYTEST_ARGS -k "not test_total_runtime")
endif()

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# disable when GPU-0 is navi2, navi3, or navi4 (no OpenMP target offload support)
list(GET rocprofiler-sdk-tests-gfx-info 0 ompt-tracing-gpu-0-gfx-info)
if("${ompt-tracing-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(IS_DISABLED ON)
endif()

# the openmp-target binary is what we trace; if it was not built, disable the test
if(NOT TARGET openmp-target)
    set(IS_DISABLED ON)
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(ompt-tracing-env
    "OMP_NUM_THREADS=2"
    "OMP_TARGET_OFFLOAD=mandatory"
    "OMP_DISPLAY_ENV=1"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# OMPT tracing test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace --hip-trace
        --kernel-trace --marker-trace --stats -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv pftrace
        otf2 rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 120
    LABELS "integration-tests;openmp-target"
    ENVIRONMENT "${ompt-tracing-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-tracing
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS}
         -k
         "not test_granular_target_filter"
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace/out_results.json
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace/out_results.otf2
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace/out_results.pftrace
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace/out_ompt_trace.csv
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace/out_results.db
         --stats-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace/out_ompt_stats.csv
    DISCOVERY_ARGS ${PYTEST_ARGS} -k "not test_granular_target_filter"
    TIMEOUT 45
    LABELS "integration-tests;openmp-target"
    FIXTURES_REQUIRED rocprofv3-test-ompt-tracing
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-granular-target
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace target -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json csv
        --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target"
    ENVIRONMENT "${ompt-tracing-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-granular-target
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-granular-target
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS}
         -k test_granular_target_filter_only_target_ops
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-granular-target-trace/out_results.json
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-granular-target-trace/out_ompt_trace.csv
    DISCOVERY_ARGS ${PYTEST_ARGS} -k test_granular_target_filter_only_target_ops
    TIMEOUT 45
    LABELS "integration-tests;openmp-target"
    FIXTURES_REQUIRED rocprofv3-test-ompt-granular-target
    DISABLED "${IS_DISABLED}")
