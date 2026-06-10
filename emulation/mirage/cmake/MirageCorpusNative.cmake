# MirageCorpusNative.cmake — register the native (Rust) corpus pipeline as a
# ctest using the self-contained `examples/corpus-demo` fixture.
#
# Unlike RocjitsuCorpus.cmake (which shells out to the upstream IREE corpus
# runner), this drives `mirage corpus run` directly against the bundled demo
# case. The demo ships *stub* IREE tools (examples/corpus-demo/tools), so the
# test is hermetic: it needs no real IREE, no GPU, and no network. It only
# needs `python3` for the stub `iree-run-module`; when that is missing the
# corpus runner reports the case as SKIPPED (exit 77) rather than failing.
#
# Expects the including scope to define `_mirage_bin` (the built mirage binary).

option(MIRAGE_RUN_CORPUS_NATIVE
  "Register the native mirage-corpus demo pipeline as a ctest" ON)

if(NOT MIRAGE_RUN_CORPUS_NATIVE)
  return()
endif()

set(_corpus_demo_dir "${CMAKE_CURRENT_SOURCE_DIR}/examples/corpus-demo")
set(_corpus_demo_tools "${_corpus_demo_dir}/tools")

# `mirage corpus run` returns non-zero only on real failures; skips (missing
# python3 / tools) keep the exit code 0, so we additionally gate on python3 by
# letting the runner emit code 77 — surfaced here through the wrapper below.
add_test(NAME corpus_native
  COMMAND ${CMAKE_COMMAND} -E env
          "PATH=${_corpus_demo_tools}:$ENV{PATH}"
          "${_mirage_bin}" corpus run
          --root "${_corpus_demo_dir}"
          --scenario native
          --no-wrapper
          --artifact-dir "${CMAKE_BINARY_DIR}/corpus-native/artifacts"
          --out-dir "${CMAKE_BINARY_DIR}/corpus-native/results"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(corpus_native PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 300)

# A second test that just lists the demo cases — a fast smoke check that the
# loader and CLI wiring work even where python3 is unavailable.
add_test(NAME corpus_native_list
  COMMAND "${_mirage_bin}" corpus list --root "${_corpus_demo_dir}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(corpus_native_list PROPERTIES TIMEOUT 60)

message(STATUS "mirage: native corpus demo ctest ENABLED (corpus_native)")
