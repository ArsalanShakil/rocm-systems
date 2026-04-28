#!/usr/bin/env python3

import argparse
import concurrent.futures
import dataclasses
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time


@dataclasses.dataclass
class RunningTest:
    test_name: str
    process: subprocess.Popen[str]


active_tests_lock = threading.Lock()
active_tests: dict[str, RunningTest] = {}
interrupt_requested = threading.Event()


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("binary", help="path to test binary")
    parser.add_argument("--ais-capable-dir", default="/tmp", metavar="DIR", help="directory to pass to tests (default: /tmp)")
    parser.add_argument("--test-timeout", type=float, metavar="SECONDS", default=None, help="kill test if it exceeds this many seconds")
    parser.add_argument("-j", type=int, default=1, metavar="N", help="number of tests to run in parallel (default: 1)")
    parser.add_argument("--silent", action="store_true", help="suppress output for passing tests")
    parser.add_argument("--log-dir", metavar="DIR", default=None, help="write each test's output to a separate file in DIR")
    parser.add_argument("--amd-log-level", type=int, choices=range(1, 6), metavar="{1-5}", default=None, help="set AMD_LOG_LEVEL environment variable (1-5)")
    parser.add_argument("--hsakmt-debug-level", type=int, choices=range(3, 8), metavar="{3-7}", default=None, help="set HSAKMT_DEBUG_LEVEL environment variable (3-7)")
    parser.add_argument("--ltrace-hsa", action="store_true", help="run each test under `ltrace -C -e \"hsa*\"`")
    return parser.parse_args()


def list_tests(binary: str) -> list[str]:
    """Return all test names (suite.test) reported by the test binary."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [binary, "--gtest_list_tests"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    tests: list[str] = []
    current_suite: str = ""

    line: str
    for line in result.stdout.splitlines():
        if not line:
            continue
        if line[0] != " ":
            # Suite name — ends with '.'
            current_suite = line.split()[0]
        elif len(line) > 2 and line[0] == " " and line[1] == " " and line[2] != " ":
            # Test names are indented exactly two spaces (column 2 = index 2)
            test_name: str = line[2:].split()[0]
            tests.append(current_suite + test_name)

    return tests


def send_signal_to_test_process(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def stop_test_process(process: subprocess.Popen[str]) -> None:
    send_signal_to_test_process(process, signal.SIGINT)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    send_signal_to_test_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    send_signal_to_test_process(process, signal.SIGKILL)
    process.wait()


def interrupt_running_tests() -> None:
    with active_tests_lock:
        running_tests = list(active_tests.values())

    for running_test in running_tests:
        stop_test_process(running_test.process)


def log_file_path(log_dir: pathlib.Path, test_name: str, test_index: int, pid: int | None) -> pathlib.Path:
    safe_test_name: str = re.sub(r"[^A-Za-z0-9._-]+", "_", test_name).strip("._")
    if not safe_test_name:
        safe_test_name = f"test_{test_index + 1}"
    pid_part: str = str(pid) if pid is not None else "unknown-pid"
    return log_dir / f"{test_index + 1:04d}_{pid_part}_{safe_test_name}.log"


def write_test_log(log_dir: pathlib.Path | None, test_name: str, test_index: int, pid: int | None, output: str) -> None:
    if log_dir is None:
        return

    log_path: pathlib.Path = log_file_path(log_dir, test_name, test_index, pid)
    log_path.write_text(output)


def run_test(binary: str, test_name: str, ais_capable_dir: str, timeout: float | None = None,
             amd_log_level: int | None = None, hsakmt_debug_level: int | None = None,
             ltrace_hsa: bool = False) -> tuple[bool, bool, bool, int | None, str]:
    """Run a single test and return (passed, timed_out, interrupted, pid, output)."""
    env: dict[str, str] = os.environ.copy()
    if amd_log_level is not None:
        env["AMD_LOG_LEVEL"] = str(amd_log_level)
    if hsakmt_debug_level is not None:
        env["HSAKMT_DEBUG_LEVEL"] = str(hsakmt_debug_level)

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as output_file:
        command: list[str] = [binary, f"--gtest_filter={test_name}", f"--ais-capable-dir={ais_capable_dir}"]
        if ltrace_hsa:
            command = ["ltrace", "-C", "-e", "hsa*"] + command

        process = subprocess.Popen(
            command,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )

        with active_tests_lock:
            active_tests[test_name] = RunningTest(test_name=test_name, process=process)
        pid: int = process.pid

        try:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_test_process(process)
                output_file.seek(0)
                return False, True, False, pid, output_file.read() + f"\n[TIMEOUT after {timeout}s]"

            output_file.seek(0)
            output: str = output_file.read()
            if interrupt_requested.is_set():
                return False, False, True, pid, output
            return returncode == 0, False, False, pid, output
        finally:
            with active_tests_lock:
                active_tests.pop(test_name, None)


def print_result(test_name: str, ok: bool, did_timeout: bool, interrupted: bool, output: str, *, silent: bool) -> tuple[int, int, int]:
    passed_delta: int = 0
    failed_delta: int = 0
    timed_out_delta: int = 0

    if ok:
        passed_delta = 1
        if not silent:
            print(f"PASS  {test_name}")
    elif did_timeout:
        timed_out_delta = 1
        print(f"TIMEOUT  {test_name}  [{time.monotonic_ns()}ns]")
        print(output)
    elif interrupted:
        print(f"INTERRUPTED  {test_name}")
        print(output)
    else:
        failed_delta = 1
        print(f"FAIL  {test_name}")
        print(output)

    return passed_delta, failed_delta, timed_out_delta


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    tests: list[str] = list_tests(args.binary)
    log_dir: pathlib.Path | None = pathlib.Path(args.log_dir) if args.log_dir is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    passed: int = 0
    failed: int = 0
    timed_out: int = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.j)
    futures: dict[concurrent.futures.Future[tuple[bool, bool, bool, int | None, str]], str] = {
        executor.submit(run_test, args.binary, test_name, args.ais_capable_dir, args.test_timeout,
                        args.amd_log_level, args.hsakmt_debug_level, args.ltrace_hsa): test_name
        for test_name in tests
    }
    test_indexes: dict[str, int] = {test_name: index for index, test_name in enumerate(tests)}
    pending_futures: dict[concurrent.futures.Future[tuple[bool, bool, bool, int | None, str]], str] = dict(futures)

    try:
        while pending_futures:
            done, _ = concurrent.futures.wait(
                pending_futures,
                timeout=0.2,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue

            future: concurrent.futures.Future[tuple[bool, bool, bool, int | None, str]]
            for future in done:
                test_name = pending_futures.pop(future)
                ok: bool
                did_timeout: bool
                interrupted: bool
                pid: int | None
                output: str
                ok, did_timeout, interrupted, pid, output = future.result()
                write_test_log(log_dir, test_name, test_indexes[test_name], pid, output)
                passed_delta: int
                failed_delta: int
                timed_out_delta: int
                passed_delta, failed_delta, timed_out_delta = print_result(
                    test_name,
                    ok,
                    did_timeout,
                    interrupted,
                    output,
                    silent=args.silent,
                )
                passed += passed_delta
                failed += failed_delta
                timed_out += timed_out_delta
    except KeyboardInterrupt:
        interrupt_requested.set()
        print("\nInterrupted. Collecting output from running tests...", file=sys.stderr)
        interrupt_running_tests()
        executor.shutdown(wait=False, cancel_futures=True)

        for future, test_name in list(pending_futures.items()):
            try:
                ok, did_timeout, interrupted, pid, output = future.result(timeout=15)
            except concurrent.futures.CancelledError:
                continue
            except concurrent.futures.TimeoutError:
                print(f"INTERRUPTED  {test_name}")
                print("[Unable to collect output before exit]")
                continue

            write_test_log(log_dir, test_name, test_indexes[test_name], pid, output)
            print_result(
                test_name,
                ok,
                did_timeout,
                interrupted or True,
                output,
                silent=args.silent,
            )
        sys.exit(130)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    print(f"\n{passed} passed, {failed} failed, {timed_out} timed out")
    exit_code: int = 0
    if failed > 0:
        exit_code |= 1
    if timed_out > 0:
        exit_code |= 2
    sys.exit(exit_code)
