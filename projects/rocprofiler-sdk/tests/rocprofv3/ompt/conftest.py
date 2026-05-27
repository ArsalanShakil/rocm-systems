#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import csv
import json
import os
import sqlite3

import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader
from rocprofiler_sdk.pytest_utils.otf2_reader import OTF2Reader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        default="ompt/openmp-target-trace/out_results.json",
        help="Input JSON",
    )
    parser.addoption(
        "--otf2-input",
        action="store",
        default="ompt/openmp-target-trace/out_results.otf2",
        help="Input OTF2",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        default="ompt/openmp-target-trace/out_results.pftrace",
        help="Input pftrace file",
    )
    parser.addoption(
        "--csv-input",
        action="store",
        default="ompt/openmp-target-trace/out_ompt_trace.csv",
        help="Input CSV",
    )
    parser.addoption(
        "--rocpd-input",
        action="store",
        default="ompt/openmp-target-trace/out_results.db",
        help="Input rocpd SQLite database",
    )
    parser.addoption(
        "--stats-input",
        action="store",
        default="ompt/openmp-target-trace/out_ompt_stats.csv",
        help="Input OMPT stats CSV",
    )
    parser.addoption(
        "--agent-input",
        action="store",
        default="ompt/openmp-target-trace/out_agent_info.csv",
        help="Input agent info CSV",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    if not os.path.isfile(filename):
        return pytest.skip("OMPT tracing unavailable")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def csv_data(request):
    filename = request.config.getoption("--csv-input")
    data = []
    if not os.path.isfile(filename):
        # The CSV file is not generated, because the dependency test
        # responsible to generate this file was skipped or failed.
        return pytest.skip("OMPT tracing unavailable")
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def otf2_data(request):
    filename = request.config.getoption("--otf2-input")
    if not os.path.isfile(filename):
        return pytest.skip("OMPT tracing unavailable")
    return OTF2Reader(filename).read()[0]


@pytest.fixture
def pftrace_data(request):
    filename = request.config.getoption("--pftrace-input")
    if not os.path.isfile(filename):
        return pytest.skip("OMPT tracing unavailable")
    return PerfettoReader(filename).read()[0]


@pytest.fixture
def rocpd_conn(request):
    """Open the rocpd SQLite DB. Skipped when --output-format rocpd was not
    requested (the combined-trace integration test enables it).
    """
    filename = request.config.getoption("--rocpd-input")
    if not os.path.isfile(filename):
        return pytest.skip("rocpd output unavailable")
    return sqlite3.connect(filename)


@pytest.fixture
def stats_data(request):
    """Return the OMPT *_ompt_stats.csv as a list of dict rows. Skipped when
    --stats was not requested.
    """
    filename = request.config.getoption("--stats-input")
    if not os.path.isfile(filename):
        return pytest.skip("OMPT stats output unavailable")
    rows = []
    with open(filename, "r") as inp:
        for row in csv.DictReader(inp):
            rows.append(row)
    return rows


@pytest.fixture
def agent_info_input_data(request):
    """Return the *_agent_info.csv as a list of dict rows. Matches the
    ``agent_info_input_data`` fixture in tests/rocprofv3/tracing/conftest.py.
    """
    filename = request.config.getoption("--agent-input")
    if not os.path.isfile(filename):
        return pytest.skip("agent_info CSV unavailable")
    rows = []
    with open(filename, "r") as inp:
        for row in csv.DictReader(inp):
            rows.append(row)
    return rows
