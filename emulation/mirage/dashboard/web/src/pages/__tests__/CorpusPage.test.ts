import { describe, expect, it } from "vitest";
import type { CorpusRunReport } from "../../api/types";
import { outcomeFor, summarize } from "../CorpusPage";

const report: CorpusRunReport = {
  outcomes: [
    {
      case: "matmul",
      scenario: "rocjitsu",
      config: "default",
      status: "pass",
      elapsed_s: 0.12,
      returncode: 0,
      message: "",
    },
    {
      case: "matmul",
      scenario: "hotswap",
      config: "default",
      status: "skip",
      elapsed_s: 0,
      returncode: 77,
      message: "emulator not installed",
    },
    {
      case: "relu",
      scenario: "rocjitsu",
      config: "default",
      status: "fail",
      elapsed_s: 0.05,
      returncode: 1,
      message: "mismatch",
    },
    {
      case: "relu",
      scenario: "native",
      config: "default",
      status: "xfail",
      elapsed_s: 0.01,
      returncode: 1,
      message: "expected failure",
    },
  ],
};

describe("outcomeFor", () => {
  it("finds the outcome for a case+scenario pair", () => {
    expect(outcomeFor(report, "matmul", "rocjitsu")?.status).toBe("pass");
    expect(outcomeFor(report, "matmul", "hotswap")?.status).toBe("skip");
  });

  it("returns undefined for an unknown pair", () => {
    expect(outcomeFor(report, "matmul", "native")).toBeUndefined();
    expect(outcomeFor(null, "matmul", "rocjitsu")).toBeUndefined();
  });
});

describe("summarize", () => {
  it("counts xfail as passed and xpass/fail as failed", () => {
    expect(summarize(report)).toEqual({ passed: 2, failed: 1, skipped: 1 });
  });

  it("handles a null report", () => {
    expect(summarize(null)).toEqual({ passed: 0, failed: 0, skipped: 0 });
  });
});
