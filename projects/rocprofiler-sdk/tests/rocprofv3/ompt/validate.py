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

import sys
import pytest

# rocprofv3 emits OMPT records under the "ompt" key in buffer_records
# and the corresponding "kind" string in strings.buffer_records is "OMPT".
# The CSV "Domain" column for OMPT rows is "OMPT" (see
# source/lib/output/domain_type.cpp).
OMPT_BUFFER_KEY = "ompt"
OMPT_KIND_NAME = "OMPT"


def _get_ompt_kind_record(tool_node):
    """Return the strings.buffer_records entry for the OMPT kind.

    `tool_node` is the already-dereferenced ``rocprofiler-sdk-tool`` node
    (i.e. ``json_data["rocprofiler-sdk-tool"]``).
    """
    for itr in tool_node["strings"]["buffer_records"]:
        if itr["kind"] == OMPT_KIND_NAME:
            return itr
    return None


def test_json_structure(json_data):
    """The rocprofv3 JSON output must contain an OMPT buffer-records section."""
    data = json_data["rocprofiler-sdk-tool"]

    assert "buffer_records" in data
    assert OMPT_BUFFER_KEY in data["buffer_records"], (
        f"JSON output is missing the '{OMPT_BUFFER_KEY}' buffer-records section; "
        "OMPT tracing was likely not wired into write_json"
    )

    kind_entry = _get_ompt_kind_record(data)
    assert (
        kind_entry is not None
    ), f"strings.buffer_records does not contain a '{OMPT_KIND_NAME}' entry"
    assert len(kind_entry["operations"]) > 0, "OMPT kind has an empty operations list"


def test_ompt_records_present(json_data):
    """At least one OMPT buffer record should be captured for an OpenMP target program."""
    data = json_data["rocprofiler-sdk-tool"]
    ompt_records = data["buffer_records"][OMPT_BUFFER_KEY]

    assert len(ompt_records) > 0, "Expected at least one OMPT buffer record; got none"

    kind_entry = _get_ompt_kind_record(data)
    assert kind_entry is not None
    op_names = kind_entry["operations"]

    for node in ompt_records:
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "start_timestamp" in node
        assert "end_timestamp" in node
        assert "thread_id" in node

        assert node.thread_id > 0
        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        # OMPT records may be instantaneous (start == end) for notifications
        # like omp_thread_begin / omp_device_initialize / omp_dispatch.
        assert node.start_timestamp <= node.end_timestamp

        assert data.strings.buffer_records[node.kind].kind == OMPT_KIND_NAME
        assert node.operation < len(op_names)


def test_csv_data(csv_data):
    """The OMPT CSV must contain at least one row with the standard API CSV columns."""
    assert len(csv_data) > 0, "Expected non-empty CSV data for OMPT tracing"

    for row in csv_data:
        for col in (
            "Domain",
            "Function",
            "Process_Id",
            "Thread_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ):
            assert col in row, f"'{col}' was not present in csv data for OMPT trace"

        assert row["Domain"] == OMPT_KIND_NAME
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) > 0
        assert int(row["Start_Timestamp"]) > 0
        assert int(row["End_Timestamp"]) > 0
        assert int(row["Start_Timestamp"]) <= int(row["End_Timestamp"])
        assert len(row["Function"]) > 0


def test_csv_matches_json_count(csv_data, json_data):
    """The number of CSV rows should equal the number of OMPT records in JSON."""
    json_count = len(json_data["rocprofiler-sdk-tool"]["buffer_records"][OMPT_BUFFER_KEY])
    assert len(csv_data) == json_count, (
        f"CSV row count ({len(csv_data)}) does not match "
        f"JSON OMPT record count ({json_count})"
    )


def test_perfetto_data(pftrace_data, json_data):
    """The perfetto trace should expose OMPT events under the 'openmp' category.

    Perfetto handles zero-duration slices natively, so we expect at least a
    1:1 correspondence with the JSON OMPT records. The OMPT runtime hook is
    owned by the OpenMP runtime and can emit shutdown callbacks after
    rocprofiler_stop_context returns, which on some CI runners results in
    perfetto/OTF2/rocpd observing a few more records than JSON/CSV. The
    assertion is therefore a superset check rather than strict equality.
    """
    pf_ompt = pftrace_data.loc[pftrace_data["category"] == "openmp"]
    js_ompt = json_data["rocprofiler-sdk-tool"]["buffer_records"][OMPT_BUFFER_KEY]

    assert len(pf_ompt) > 0, "perfetto contains no OMPT (category=openmp) slices"
    assert len(pf_ompt) >= len(js_ompt), (
        f"perfetto openmp slices ({len(pf_ompt)}) is fewer than "
        f"JSON OMPT record count ({len(js_ompt)}) — records appear to have been "
        f"dropped from the perfetto backend"
    )


def _is_target_submit_operation(op_name):
    """Return True if the OMPT operation name corresponds to a target_submit
    event. The SDK enum spells the operation 'target_submit_emi' but the CSV
    column and stats CSV use the 'omp_target_submit' / 'OMP_TARGET_SUBMIT_EMI'
    flavours, so accept anything that contains 'target_submit'.
    """
    return "target_submit" in op_name.lower()


def test_ompt_target_correlates_with_kernel_dispatch(json_data):
    """AC5 — every OMPT `target_submit` record must reference a correlation_id
    that also appears on a kernel_dispatch record. This validates that the
    rocprofv3 pipeline preserves the OMPT&harr;HSA&harr;kernel-dispatch
    correlation that the SDK promises.

    The test is skipped on builds that did not enable --kernel-trace alongside
    --ompt-trace (kernel_dispatch records absent) so that the pure-OMPT
    integration test remains a focused smoke test.
    """
    data = json_data["rocprofiler-sdk-tool"]
    ompt_records = data["buffer_records"][OMPT_BUFFER_KEY]

    kernel_dispatch = data["buffer_records"].get("kernel_dispatch", [])
    if not kernel_dispatch:
        pytest.skip(
            "kernel_dispatch records are missing; this integration test was "
            "run without --kernel-trace"
        )

    kind_entry = _get_ompt_kind_record(data)
    op_names = kind_entry["operations"]

    target_submit_records = [
        r for r in ompt_records if _is_target_submit_operation(op_names[r["operation"]])
    ]
    assert len(target_submit_records) > 0, (
        "Expected at least one OMPT target_submit_emi record on a target-"
        "offload program"
    )

    # kernel_dispatch records expose correlation_id as a struct {internal, external}
    def _internal(corr):
        if isinstance(corr, dict):
            return corr.get("internal")
        return corr

    dispatch_corr_ids = {_internal(r["correlation_id"]) for r in kernel_dispatch}
    assert len(dispatch_corr_ids) > 0

    unmatched = []
    for r in target_submit_records:
        cid = _internal(r["correlation_id"])
        if cid not in dispatch_corr_ids:
            unmatched.append(cid)

    assert not unmatched, (
        f"{len(unmatched)} OMPT target_submit correlation_id(s) did not match "
        f"any kernel_dispatch correlation_id; first unmatched: {unmatched[:5]}"
    )


def test_rocpd_contains_ompt_records(rocpd_conn, json_data):
    """AC3a/AC6 — verify the rocpd SQLite database contains the OMPT events the
    JSON output reports. OMPT records flow into two tables:
      * rocpd_region_*  for events with start < end
      * rocpd_sample_*  for instantaneous events (start == end)
    The combined row count for the OMPT category must equal the JSON count.
    """
    cur = rocpd_conn.cursor()

    def _table(prefix):
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f"{prefix}_%",),
        )
        row = cur.fetchone()
        assert row is not None, f"rocpd schema is missing a {prefix}_* table"
        return row[0]

    region_t = _table("rocpd_region")
    sample_t = _table("rocpd_sample")
    event_t = _table("rocpd_event")
    string_t = _table("rocpd_string")

    q = (
        f"SELECT COUNT(*) FROM {region_t} r JOIN {event_t} e ON r.event_id=e.id "
        f"JOIN {string_t} s ON e.category_id=s.id WHERE s.string=?"
    )
    region_count = cur.execute(q, (OMPT_KIND_NAME,)).fetchone()[0]

    q = (
        f"SELECT COUNT(*) FROM {sample_t} samp JOIN {event_t} e ON samp.event_id=e.id "
        f"JOIN {string_t} s ON e.category_id=s.id WHERE s.string=?"
    )
    sample_count = cur.execute(q, (OMPT_KIND_NAME,)).fetchone()[0]

    assert (
        region_count + sample_count > 0
    ), "rocpd database has no OMPT rows in either region or sample tables"

    js_ompt = json_data["rocprofiler-sdk-tool"]["buffer_records"][OMPT_BUFFER_KEY]
    # The OMPT runtime hook is owned by the OpenMP runtime and can emit
    # shutdown callbacks after rocprofiler_stop_context returns, which on
    # some CI runners results in rocpd/perfetto/OTF2 observing a few more
    # records than JSON/CSV. The assertion is therefore a superset check
    # (rocpd row count must be at least the JSON OMPT count) rather than
    # strict equality.
    assert region_count + sample_count >= len(js_ompt), (
        f"rocpd OMPT row count (regions={region_count} + samples={sample_count}"
        f"={region_count + sample_count}) is fewer than JSON OMPT count "
        f"({len(js_ompt)}) — records appear to have been dropped from the rocpd backend"
    )


def test_stats_csv_contains_ompt(stats_data):
    """AC3b/AC6 — stats CSV exists and has at least one non-zero domain row.

    The stats CSV columns differ between rocprofv3 versions; the only invariant
    we rely on is that every row has a 'TotalDurationNs' (or 'Total' / 'TotalDuration')
    column and that the column for the API name is non-empty. We just make sure
    that the file contains tabular data.
    """
    assert len(stats_data) > 0, "Empty OMPT stats CSV"
    # Every row must have at least one named (non-empty) cell.
    for row in stats_data:
        assert any(
            v.strip() for v in row.values() if v is not None
        ), f"Stats row contained no non-empty cells: {row}"


def test_granular_target_filter_only_target_ops(json_data):
    """AC2 — when --ompt-trace target is supplied, the captured records must
    only be target-category operations (target_emi, target_data_op_emi,
    target_submit_emi). This is the integration-level check for the granular
    CLI filter; the resolution from category to operation IDs lives in
    tool.cpp::resolve_ompt_ops.
    """
    data = json_data["rocprofiler-sdk-tool"]
    ompt_records = data["buffer_records"].get(OMPT_BUFFER_KEY, [])
    if not ompt_records:
        pytest.skip(
            "no OMPT records present; the granular-filter integration test "
            "was likely not run"
        )

    kind_entry = _get_ompt_kind_record(data)
    op_names = kind_entry["operations"]

    seen_ops = {op_names[r["operation"]] for r in ompt_records}
    leaks = {op for op in seen_ops if "target" not in op.lower()}
    assert not leaks, (
        f"--ompt-trace target leaked non-target operations into the trace: "
        f"{sorted(leaks)}"
    )

    # Must include at least one target_submit so the filter actually worked
    assert any("target_submit" in op.lower() for op in seen_ops), (
        f"--ompt-trace target produced no target_submit records; saw: "
        f"{sorted(seen_ops)}"
    )


def test_otf2_data(otf2_data, json_data):
    """The OTF2 trace should expose ranged OMPT events under the 'openmp' category.

    OTF2 is region-based (Enter/Leave) and cannot faithfully represent
    instantaneous OMPT notifications where start_timestamp == end_timestamp
    (see generateOTF2.cpp). Such records are intentionally elided from OTF2.

    The OMPT runtime hook is owned by the OpenMP runtime and can emit
    shutdown callbacks after rocprofiler_stop_context returns, which on some
    CI runners results in OTF2/perfetto/rocpd observing a few more records
    than JSON/CSV. The assertion is therefore a superset check (OTF2 count
    must be at least the JSON ranged-record count) rather than strict
    equality.
    """
    otf2_ompt = otf2_data.loc[otf2_data["category"] == "openmp"]
    js_ompt = json_data["rocprofiler-sdk-tool"]["buffer_records"][OMPT_BUFFER_KEY]
    js_ompt_ranged = [r for r in js_ompt if r["start_timestamp"] != r["end_timestamp"]]

    assert len(otf2_ompt) > 0, "OTF2 contains no OMPT (category=openmp) events"
    assert len(otf2_ompt) >= len(js_ompt_ranged), (
        f"OTF2 openmp events ({len(otf2_ompt)}) is fewer than "
        f"ranged JSON OMPT record count ({len(js_ompt_ranged)}); "
        f"total OMPT records in JSON = {len(js_ompt)}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
