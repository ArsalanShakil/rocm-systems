#!/usr/bin/env bash
# Mirage corpus demo: run the bundled `relu_f32` case across scenarios using
# stub IREE tools, so it works with no real IREE install and no GPU.
#
#   ./demo.sh                  # run with the mirage on your PATH
#   MIRAGE=/path/to/mirage ./demo.sh
#
# What it shows:
#   * discovering corpus cases (TOML),
#   * CEL-driven input generation + CEL output validation,
#   * running the same kernel under the `native` scenario,
#   * the pass/fail results matrix.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mirage="${MIRAGE:-mirage}"

# Put the stub iree-compile / iree-run-module first on PATH.
export PATH="$here/tools:$PATH"

echo "== scenarios =="
"$mirage" corpus scenarios

echo
echo "== cases under $here =="
"$mirage" corpus list --root "$here"

echo
echo "== show relu_f32 =="
"$mirage" corpus show --root "$here" relu_f32

echo
echo "== run (native scenario, no wrapper) =="
"$mirage" corpus run \
  --root "$here" \
  --scenario native \
  --no-wrapper \
  --artifact-dir "$here/.artifacts" \
  --out-dir "$here/.results"

echo
echo "results CSV: $here/.results/results.csv"
cat "$here/.results/results.csv"
