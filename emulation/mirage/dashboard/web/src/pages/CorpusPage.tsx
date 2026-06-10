import { useEffect, useState } from "react";
import * as api from "../api/client";
import type {
  CaseOutcome,
  CaseStatus,
  CorpusCase,
  CorpusRunReport,
  CorpusScenario,
} from "../api/types";

/// Look up the outcome for a given case + scenario in a report.
export function outcomeFor(
  report: CorpusRunReport | null,
  caseName: string,
  scenario: string,
): CaseOutcome | undefined {
  return report?.outcomes.find(
    (o) => o.case === caseName && o.scenario === scenario,
  );
}

/// Summarise a report as `{passed, failed, skipped}`.
export function summarize(report: CorpusRunReport | null): {
  passed: number;
  failed: number;
  skipped: number;
} {
  const out = { passed: 0, failed: 0, skipped: 0 };
  for (const o of report?.outcomes ?? []) {
    if (o.status === "pass" || o.status === "xfail") out.passed += 1;
    else if (o.status === "fail" || o.status === "xpass") out.failed += 1;
    else out.skipped += 1;
  }
  return out;
}

const STATUS_COLORS: Record<CaseStatus, string> = {
  pass: "#1a7f37",
  fail: "#cf222e",
  skip: "#6e7781",
  xfail: "#9a6700",
  xpass: "#cf222e",
};

export function CorpusPage() {
  const [root, setRoot] = useState("");
  const [scenarios, setScenarios] = useState<CorpusScenario[]>([]);
  const [selectedScenarios, setSelectedScenarios] = useState<Set<string>>(
    new Set(),
  );
  const [cases, setCases] = useState<CorpusCase[]>([]);
  const [selectedCases, setSelectedCases] = useState<Set<string>>(new Set());
  const [report, setReport] = useState<CorpusRunReport | null>(null);
  const [compileOnly, setCompileOnly] = useState(false);
  const [bench, setBench] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listCorpusScenarios()
      .then((s) => {
        setScenarios(s);
        setSelectedScenarios(new Set(s.map((x) => x.name)));
      })
      .catch((e) => setError(String(e)));
  }, []);

  async function loadCases() {
    if (!root.trim()) {
      setError("Enter a corpus root directory first.");
      return;
    }
    setError("");
    setReport(null);
    try {
      const c = await api.listCorpusCases(root.trim());
      setCases(c);
      setSelectedCases(new Set(c.map((x) => x.name)));
    } catch (e) {
      setError(String(e));
    }
  }

  function toggle(set: Set<string>, key: string): Set<string> {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  }

  async function run() {
    if (!root.trim()) {
      setError("Enter a corpus root directory first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const r = await api.runCorpus({
        root: root.trim(),
        cases: cases.length === selectedCases.size ? [] : [...selectedCases],
        scenarios: [...selectedScenarios],
        compile_only: compileOnly,
      });
      setReport(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const activeScenarios = scenarios.filter((s) =>
    selectedScenarios.has(s.name),
  );
  const shownCases = cases.filter((c) => selectedCases.has(c.name));
  const summary = summarize(report);

  return (
    <div className="page">
      <header className="page-header">
        <h2>Corpus</h2>
        <p className="page-subtitle">
          Run rocjitsu test-corpus kernels across emulator scenarios.
        </p>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <section className="card">
        <h3>Corpus root</h3>
        <div className="row">
          <input
            type="text"
            placeholder="/path/to/rocjitsu-test-corpus"
            value={root}
            onChange={(e) => setRoot(e.target.value)}
            aria-label="corpus-root"
            style={{ flex: 1 }}
          />
          <button onClick={loadCases} data-testid="load-cases">
            Discover cases
          </button>
        </div>
      </section>

      <section className="card">
        <h3>Scenarios</h3>
        <div className="checkbox-grid">
          {scenarios.map((s) => (
            <label key={s.name} className="checkbox-row" title={s.description}>
              <input
                type="checkbox"
                checked={selectedScenarios.has(s.name)}
                onChange={() =>
                  setSelectedScenarios((set) => toggle(set, s.name))
                }
              />
              <span>
                {s.name}
                {!s.installed && (
                  <em style={{ color: "#6e7781" }}> (not installed)</em>
                )}
              </span>
            </label>
          ))}
        </div>
      </section>

      {cases.length > 0 && (
        <section className="card">
          <h3>Cases ({cases.length})</h3>
          <div className="checkbox-grid">
            {cases.map((c) => (
              <label key={c.name} className="checkbox-row" title={c.function}>
                <input
                  type="checkbox"
                  checked={selectedCases.has(c.name)}
                  onChange={() =>
                    setSelectedCases((set) => toggle(set, c.name))
                  }
                />
                <span>{c.name}</span>
              </label>
            ))}
          </div>
        </section>
      )}

      <section className="card">
        <div className="row">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={compileOnly}
              onChange={(e) => setCompileOnly(e.target.checked)}
            />
            <span>Compile only</span>
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={bench}
              onChange={(e) => setBench(e.target.checked)}
            />
            <span>Show timings</span>
          </label>
          <button
            onClick={run}
            disabled={busy}
            data-testid="run-corpus"
            className="primary"
          >
            {busy ? "Running…" : "Run"}
          </button>
        </div>
      </section>

      {report && (
        <section className="card">
          <h3>Results</h3>
          <p data-testid="run-summary">
            {summary.passed} passed, {summary.failed} failed, {summary.skipped}{" "}
            skipped
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th>Case</th>
                {activeScenarios.map((s) => (
                  <th key={s.name}>{s.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shownCases.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  {activeScenarios.map((s) => {
                    const o = outcomeFor(report, c.name, s.name);
                    return (
                      <td
                        key={s.name}
                        title={o?.message ?? ""}
                        style={{
                          color: o ? STATUS_COLORS[o.status] : "#6e7781",
                          fontWeight: 600,
                        }}
                      >
                        {o
                          ? bench && o.status === "pass"
                            ? `${o.elapsed_s.toFixed(3)}s`
                            : o.status
                          : "-"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
