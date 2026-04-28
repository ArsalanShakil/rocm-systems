#!/usr/bin/env python3

import argparse
import datetime
import os
import pathlib
import signal
import subprocess
import sys
import uuid


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("binary", help="path to test binary")
    parser.add_argument("--ais-capable-dir", default="/tmp", metavar="DIR", help="directory to pass to tests (default: /tmp)")
    parser.add_argument("--test-timeout", type=float, metavar="SECONDS", default=None, help="kill test if it exceeds this many seconds")
    parser.add_argument("-j", type=int, default=1, metavar="N", help="number of tests to run in parallel (default: 1)")
    parser.add_argument("--silent", action="store_true", help="suppress output for passing tests")
    parser.add_argument("--write-test-logs", action="store_true", help="write per-test logs for each run under repeat-test-log-<uuid>/")
    parser.add_argument("--fail-fast", action="store_true", help="exit on first timeout or failure")
    parser.add_argument("--amd-log-level", type=int, choices=range(1, 6), metavar="{1-5}", default=None, help="set AMD_LOG_LEVEL environment variable (1-5)")
    parser.add_argument("--hsakmt-debug-level", type=int, choices=range(3, 8), metavar="{3-7}", default=None, help="set HSAKMT_DEBUG_LEVEL environment variable (3-7)")
    parser.add_argument("--ltrace-hsa", action="store_true", help="run each test under `ltrace -C -e \"hsa*\"`")
    return parser.parse_args()


def save_output(output_dir: pathlib.Path, run_index: int, output: str) -> pathlib.Path:
    output_dir.mkdir(exist_ok=True)
    log_path: pathlib.Path = output_dir / str(run_index)
    log_path.write_text(output)
    return log_path


def interrupt_and_collect(process: subprocess.Popen[str]) -> str:
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass

    try:
        output, _ = process.communicate(timeout=5)
        return output
    except subprocess.TimeoutExpired:
        process.terminate()

    try:
        output, _ = process.communicate(timeout=5)
        return output
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        return output


def build_run_tests_command(args: argparse.Namespace, log_dir: pathlib.Path | None) -> list[str]:
    command: list[str] = [
        sys.executable,
        str(pathlib.Path(__file__).parent / "run_tests.py"),
        args.binary,
        "--ais-capable-dir",
        args.ais_capable_dir,
        "-j",
        str(args.j),
    ]
    if args.test_timeout is not None:
        command.extend(["--test-timeout", str(args.test_timeout)])
    if args.silent:
        command.append("--silent")
    if args.amd_log_level is not None:
        command.extend(["--amd-log-level", str(args.amd_log_level)])
    if args.hsakmt_debug_level is not None:
        command.extend(["--hsakmt-debug-level", str(args.hsakmt_debug_level)])
    if args.ltrace_hsa:
        command.append("--ltrace-hsa")
    if log_dir is not None:
        command.extend(["--log-dir", str(log_dir)])
    return command


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()  # validate args before starting the loop
    binary: pathlib.Path = pathlib.Path(args.binary)
    if not binary.exists():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)
    if not os.access(binary, os.X_OK):
        print(f"error: binary is not executable: {binary}", file=sys.stderr)
        sys.exit(1)
    run_id: str = str(uuid.uuid4())
    run_timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root_name: str = f"repeat-test-log-{run_timestamp}-{run_id}"
    output_dir: pathlib.Path = pathlib.Path(output_root_name)
    test_log_root: pathlib.Path | None = None
    if args.write_test_logs:
        test_log_root = output_dir
        test_log_root.mkdir(parents=True, exist_ok=True)
    run_index: int = 0
    success: int = 0
    timed_out: int = 0
    errored: int = 0

    while True:
        run_index += 1
        run_log_dir: pathlib.Path | None = None
        if test_log_root is not None:
            run_log_dir = test_log_root / f"run-{run_index:04d}"
            run_log_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            build_run_tests_command(args, run_log_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate()
        except KeyboardInterrupt:
            print(f"\nInterrupted during run {run_index}. Collecting output from active tests...", file=sys.stderr)
            stdout = interrupt_and_collect(process)
            log_path = save_output(output_dir, run_index, stdout)
            print(f"Partial output saved to {log_path}", file=sys.stderr)
            if stdout:
                print(stdout, end="")
            sys.exit(130)

        result_code: int = process.returncode if process.returncode is not None else 1
        if result_code != 0:
            timed_out += 1 if (result_code & 2) > 0 else 0
            errored += 1 if (result_code & 1) > 0 else 0
            log_path = save_output(output_dir, run_index, stdout)
            print(f"Output saved to {log_path}")
        else:
            success += 1
        print(f"Run {run_index} finished. [{success} success, {errored} error, {timed_out} timed out]")
        print(stdout, end="")
        if args.fail_fast and result_code != 0:
            sys.exit(result_code)
