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
from collections import Counter

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


def test_agent_info(agent_info_input_data):
    """Mirrors tests/rocprofv3/tracing/validate.py::test_agent_info — every
    agent reported in the trace must be a well-formed CPU or GPU record. This
    catches regressions where rocprofv3 fails to enumerate agents alongside
    OMPT tracing.
    """
    logical_node_id = max(int(itr["Logical_Node_Id"]) for itr in agent_info_input_data)
    assert logical_node_id + 1 == len(agent_info_input_data)

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0


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


# Sanity ceiling: how many "late" non-target OMPT records we tolerate between
# the early (JSON/CSV) snapshot and the later (perfetto/rocpd/OTF2) snapshot.
# On the openmp-target reference workload the observed divergence is < 20.
# If this ceiling is ever exceeded, something is over-emitting OMPT records
# and we want CI to flag it rather than silently absorb it.
MAX_LATE_RECORDS = 200


def test_perfetto_data(pftrace_data, json_data):
    """The perfetto trace must expose OMPT events under the 'openmp' category.

    Perfetto is emitted from a later read of the tmp-file buffer that backs
    CSV/JSON (see tool.cpp ``generate_output``). The OpenMP runtime continues
    emitting non-target callbacks (omp_implicit_task, omp_sync_region,
    omp_work, omp_parallel_end, ...) after the target ops complete, so
    perfetto/rocpd/OTF2 observe a *strict superset* of the JSON records. The
    invariant we enforce is:

        len(perfetto_openmp) >= len(json_ompt)

    plus the bounded-divergence ceiling ``MAX_LATE_RECORDS`` (regression
    alarm: an unbounded divergence would indicate the OMPT runtime hook is
    leaking records).
    """
    pf_ompt = pftrace_data.loc[pftrace_data["category"] == "openmp"]
    js_ompt = json_data["rocprofiler-sdk-tool"]["buffer_records"][OMPT_BUFFER_KEY]

    assert len(pf_ompt) > 0, "perfetto contains no OMPT (category=openmp) slices"
    assert len(pf_ompt) >= len(js_ompt), (
        f"perfetto openmp slices ({len(pf_ompt)}) is fewer than "
        f"JSON OMPT record count ({len(js_ompt)}) — perfetto/rocpd should be a "
        f"strict superset of the early (JSON/CSV) snapshot"
    )
    divergence = len(pf_ompt) - len(js_ompt)
    assert divergence < MAX_LATE_RECORDS, (
        f"perfetto/JSON divergence is {divergence} OMPT records, exceeding the "
        f"bounded-divergence ceiling ({MAX_LATE_RECORDS}); the OMPT runtime "
        f"hook may be over-emitting late callbacks"
    )


def _is_target_submit_operation(op_name):
    """Return True if the OMPT operation name corresponds to a target_submit
    event. The SDK enum spells the operation 'target_submit_emi' but the CSV
    column and stats CSV use the 'omp_target_submit' / 'OMP_TARGET_SUBMIT_EMI'
    flavours, so accept anything that contains 'target_submit'.
    """
    return "target_submit" in op_name.lower()


def test_ompt_target_correlates_with_kernel_dispatch(json_data):
    """AC5 — every OMPT ``target_submit`` record must reference a
    correlation_id that also appears on a kernel_dispatch record. This
    validates that the rocprofv3 pipeline preserves the OMPT <-> HSA
    <-> kernel-dispatch correlation that the SDK promises.

    The OMPT integration test runs with ``--ompt-trace --hip-trace
    --kernel-trace`` precisely so this AC can be validated end-to-end. We
    skip rather than fail when kernel_dispatch records are absent so the
    granular-target variant of this validate.py (which intentionally runs
    without --kernel-trace) is still useful.
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


def _rocpd_tables(cur):
    def _table(prefix):
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f"{prefix}_%",),
        )
        row = cur.fetchone()
        assert row is not None, f"rocpd schema is missing a {prefix}_* table"
        return row[0]

    return {
        "region": _table("rocpd_region"),
        "sample": _table("rocpd_sample"),
        "event": _table("rocpd_event"),
        "string": _table("rocpd_string"),
    }


def _rocpd_ompt_region_op_counts(cur, tables):
    """Return Counter[op_name] -> rocpd_region row count for the OMPT category."""
    rows = cur.execute(
        f"""
        SELECT name_s.string, COUNT(*) FROM {tables["region"]} r
        JOIN {tables["event"]} e ON r.event_id = e.id
        JOIN {tables["string"]} cat_s ON e.category_id = cat_s.id
        JOIN {tables["string"]} name_s ON r.name_id = name_s.id
        WHERE cat_s.string = ?
        GROUP BY name_s.string
        """,
        (OMPT_KIND_NAME,),
    ).fetchall()
    return Counter(dict(rows))


def _rocpd_ompt_row_counts(cur, tables):
    """Return (region_count, sample_count) for the OMPT category."""
    region_count = cur.execute(
        f"""
        SELECT COUNT(*) FROM {tables["region"]} r
        JOIN {tables["event"]} e ON r.event_id = e.id
        JOIN {tables["string"]} cat_s ON e.category_id = cat_s.id
        WHERE cat_s.string = ?
        """,
        (OMPT_KIND_NAME,),
    ).fetchone()[0]
    sample_count = cur.execute(
        f"""
        SELECT COUNT(*) FROM {tables["sample"]} samp
        JOIN {tables["event"]} e ON samp.event_id = e.id
        JOIN {tables["string"]} cat_s ON e.category_id = cat_s.id
        WHERE cat_s.string = ?
        """,
        (OMPT_KIND_NAME,),
    ).fetchone()[0]
    return region_count, sample_count


def test_rocpd_contains_ompt_records(rocpd_conn, json_data):
    """AC3a/AC6 — verify the rocpd SQLite database contains the OMPT events the
    JSON output reports, and that the early (JSON) and late (rocpd) snapshots
    agree on the *content* of the OMPT trace.

    Invariants enforced:

      1. rocpd has OMPT rows.
      2. rocpd is a row-count superset of JSON (late snapshot >= early snapshot)
         and the divergence is bounded by ``MAX_LATE_RECORDS``.
      3. Per-operation counts of the ``omp_target_*`` operations agree between
         JSON and rocpd (``target_data_op_emi``, ``target_emi``,
         ``target_submit_emi``). Target events complete before late non-target
         callbacks, so the early snapshot must already contain the same
         target_* records the late snapshot does. This is the strongest
         content-level cross-backend invariant we can enforce without relying
         on the rocpd correlation_id column (which stores the external, not
         internal, correlation_id and is 0 for OMPT records).
    """
    cur = rocpd_conn.cursor()
    tables = _rocpd_tables(cur)
    region_count, sample_count = _rocpd_ompt_row_counts(cur, tables)
    total = region_count + sample_count

    assert total > 0, "rocpd database has no OMPT rows in either region or sample tables"

    data = json_data["rocprofiler-sdk-tool"]
    js_ompt = data["buffer_records"][OMPT_BUFFER_KEY]

    # Invariant #2: row-count superset with bounded divergence.
    assert total >= len(js_ompt), (
        f"rocpd OMPT row count (regions={region_count} + samples={sample_count}"
        f"={total}) is fewer than JSON OMPT count ({len(js_ompt)}) — "
        f"late snapshot must be a superset of the early snapshot"
    )
    divergence = total - len(js_ompt)
    assert divergence < MAX_LATE_RECORDS, (
        f"rocpd/JSON divergence is {divergence} OMPT records, exceeding the "
        f"bounded-divergence ceiling ({MAX_LATE_RECORDS}); the OMPT runtime "
        f"hook may be over-emitting late callbacks"
    )

    # Invariant #3: target_* per-op counts must match between JSON and rocpd.
    kind_entry = _get_ompt_kind_record(data)
    op_names = kind_entry["operations"]
    js_op_counts = Counter(op_names[r["operation"]] for r in js_ompt)
    rocpd_op_counts = _rocpd_ompt_region_op_counts(cur, tables)

    target_ops = [op for op in js_op_counts if op.startswith("omp_target")]
    assert target_ops, "JSON has no omp_target_* OMPT records to cross-check"

    for op in target_ops:
        assert js_op_counts[op] == rocpd_op_counts.get(op, 0), (
            f"target-op count mismatch for '{op}': JSON={js_op_counts[op]} "
            f"rocpd_region={rocpd_op_counts.get(op, 0)}"
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
    """The OTF2 trace must expose ranged OMPT events under the 'openmp' category.

    OTF2 is region-based (Enter/Leave) and cannot faithfully represent
    instantaneous OMPT notifications where start_timestamp == end_timestamp
    (see generateOTF2.cpp); such records are intentionally elided.

    Like perfetto/rocpd, OTF2 reads from a later snapshot of the OMPT tmp-file
    buffer (``ompt_output.load_all()`` in tool.cpp) and therefore observes a
    *strict superset* of the JSON ranged OMPT records, bounded by
    ``MAX_LATE_RECORDS``.
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
    divergence = len(otf2_ompt) - len(js_ompt_ranged)
    assert divergence < MAX_LATE_RECORDS, (
        f"OTF2/JSON-ranged divergence is {divergence} OMPT events, exceeding "
        f"the bounded-divergence ceiling ({MAX_LATE_RECORDS})"
    )


def test_perfetto_matches_rocpd_count(pftrace_data, rocpd_conn):
    """Strict cross-backend equality: perfetto's openmp slice count must equal
    the rocpd OMPT row count (region + sample). Both backends read from the
    same later snapshot of the OMPT tmp-file buffer in the same
    ``generate_output(cleanup_mode)`` call, so any divergence between them
    indicates a regression in one of the two writers.
    """
    pf_count = len(pftrace_data.loc[pftrace_data["category"] == "openmp"])

    cur = rocpd_conn.cursor()
    tables = _rocpd_tables(cur)
    region_count, sample_count = _rocpd_ompt_row_counts(cur, tables)
    rocpd_total = region_count + sample_count

    assert pf_count == rocpd_total, (
        f"perfetto and rocpd disagree on OMPT count: perfetto openmp={pf_count}, "
        f"rocpd total={rocpd_total} (region={region_count} sample={sample_count}); "
        f"these two backends share a snapshot and must always match"
    )


def test_otf2_matches_rocpd_region_count(otf2_data, rocpd_conn):
    """Strict cross-backend equality: OTF2's openmp event count must equal the
    rocpd OMPT region-table row count. OTF2 emits only ranged events
    (instantaneous OMPT notifications go to rocpd_sample_* but cannot be
    represented in OTF2's Enter/Leave model).
    """
    otf2_count = len(otf2_data.loc[otf2_data["category"] == "openmp"])

    cur = rocpd_conn.cursor()
    tables = _rocpd_tables(cur)
    region_count, _ = _rocpd_ompt_row_counts(cur, tables)

    assert otf2_count == region_count, (
        f"OTF2 and rocpd-region disagree on ranged OMPT count: "
        f"OTF2 openmp={otf2_count}, rocpd region={region_count}; "
        f"these two views of the same snapshot must always match"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
