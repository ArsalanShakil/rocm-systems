# mirage corpus

`mirage corpus` runs [rocjitsu-test-corpus][corpus]–style kernel tests across
emulator **scenarios** so you can check a kernel under several backends
(rocjitsu CPU model, HotSwap on real hardware, or the native host runtime)
from one command, a REST API, or the dashboard.

A *case* describes one kernel test: the MLIR source(s) to compile, the entry
function, how to generate inputs, and how to validate outputs. Cases are plain
text — TOML (native) or the upstream IREE JSON — and inputs/outputs are
expressed with [CEL][cel] (Common Expression Language) formulas, so a case is
fully reproducible and self-describing.

```text
mirage corpus scenarios
mirage corpus list  --root <dir>
mirage corpus show  --root <dir> <case>
mirage corpus run   --root <dir> [--scenario S]... [--case C]... [flags]
mirage corpus bench --root <dir> [--scenario S]...
```

The pipeline mirrors the upstream IREE flow: each source is compiled with
`iree-compile` into a content-addressed `.vmfb` (cached by source + flag hash),
then `iree-run-module` is invoked — optionally wrapped by
`mirage run --profile <scenario> --` — and the observed `.npy` outputs are
validated. `iree-compile` and `iree-run-module` are expected on your `PATH`.

> Try it now without a real IREE install: the bundled
> [`examples/corpus-demo`](../examples/corpus-demo) ships stub IREE tools.
> Run `examples/corpus-demo/demo.sh`.

## Scenarios

A scenario binds an emulator backend + a default agent to a generated mirage
profile named `corpus-<scenario>`.

| Scenario   | Emulator   | Default agent | Needs a GPU | Notes                              |
| ---------- | ---------- | ------------- | ----------- | ---------------------------------- |
| `rocjitsu` | `rocjitsu` | MI300X        | no          | CPU functional model               |
| `hotswap`  | `hotswap`  | MI450X        | yes         | ISA rewriter on a real AMD GPU     |
| `native`   | `noop`     | MI300X        | no          | pass-through on the host runtime    |

```sh
mirage corpus scenarios          # table
mirage --json corpus scenarios   # machine-readable (includes "installed")
```

When a scenario's emulator isn't installed, its cases are reported as
**skipped** (never failed). The first time a scenario runs, mirage imports its
`corpus-<scenario>` profile automatically.

## Writing a case (TOML)

```toml
# matmul.toml
name = "matmul_f32"          # unique case name
kind = "run_module"          # only "run_module" is supported
sources = ["matmul.mlir"]    # compiled, in order, to .vmfb modules
function = "main"            # entry function

# Optional knobs (shown with defaults):
seed = 0                     # base RNG seed for input generation
rtol = 1e-5                  # relative tolerance for allclose checks
atol = 1e-8                  # absolute tolerance for allclose checks
compile_flags = []           # extra iree-compile flags for this case
run_flags = []               # extra iree-run-module flags for this case
compile_only = false         # compile but never run
vmfb_names = []              # override module filenames (default: <source>.vmfb)

# Free-form parameters exposed to every CEL expression as variables.
[params]
m = 64
n = 64
k = 32

# Output files the kernel writes (passed as --output=@file).
# IMPORTANT (TOML): all root-level keys must appear BEFORE the first
# [[inputs]] array-of-tables, or TOML folds them into that table.
outputs = ["output0.npy"]

# Validate the observed output(s) with CEL predicates (see "Validation").
validate = [
  "all_finite",
  "shape == [64, 64]",
]

# Inputs, in argument order. Each input is one of: a native CEL `formula`,
# an upstream `Formula`, or a `file`/`File` reference.
[[inputs]]
dtype = "f32"
formula = '{"dist": "uniform", "shape": [64, 32], "low": -1.0, "high": 1.0}'

[[inputs]]
dtype = "f32"
formula = '{"dist": "uniform", "shape": [32, 64], "low": -1.0, "high": 1.0}'
```

### Upstream JSON cases

The same loader reads upstream rocjitsu-test-corpus JSON cases unchanged
(`inputs[].Formula`, `inputs[].File`, `expected_output[]`, `checks[]`,
`compile_flags`, `rtol`/`atol`, …). Point `--root` at a corpus checkout and
they are discovered alongside any TOML cases. Files under `configs/` and
`schemas/` and any dot-directory (e.g. `.artifacts`) are ignored during
discovery.

### Target configs

A target config supplies shared compile/run flags and skip/xfail lists. Pass
one or more with `--config <file.json>`; when omitted a permissive default is
used.

```json
{
  "config_name": "gfx1250",
  "iree_compile_flags": ["--iree-hal-target-backends=rocm"],
  "iree_run_module_flags": ["--device=hip"],
  "iree_run_module_wrapper": [],
  "skip_compile_tests": [],
  "expected_compile_failures": [],
  "skip_run_tests": [],
  "expected_run_failures": []
}
```

`expected_compile_failures` / `expected_run_failures` turn a failure into a
pass (**xfail**) and an unexpected success into a failure (**xpass**).

## Input generation (CEL)

A native `formula` is a CEL expression that evaluates to a **generator spec**
map, which is sampled deterministically from `seed` (mixed with the input
index, so distinct inputs get distinct reproducible streams). All `[params]`
are in scope as variables.

```cel
{"dist": "uniform", "shape": [params_m, params_k], "low": -1.0, "high": 1.0}
```

Supported generator specs:

| `dist`             | Required keys        | Result                                  |
| ------------------ | -------------------- | --------------------------------------- |
| `uniform`          | `shape`,`low`,`high` | i.i.d. `U[low, high)`                    |
| `normal`           | `shape`,`mean`,`std` | i.i.d. `N(mean, std)`                    |
| `splat` / `const`  | `shape`,`value`      | every element = `value`                  |
| `zeros` / `ones`   | `shape`              | all 0 / all 1                            |
| `arange`           | `shape`,`start`,`step` | `start + step*i`                        |
| `eye`              | `n` (or `shape[0]`)  | `n×n` identity                          |

An optional `dtype` in the spec overrides the input's `dtype`. Supported
dtypes: `f32`/`f64`, `i8`/`i16`/`i32`/`i64`, `u8`/`u32`/`u64`, `bool`
(also the NumPy names `float32`… and `.npy` descrs `<f4`…). `bf16`/`f16` are
not supported for generation.

The **upstream** `Formula` form (`{dtype, shape, coeff, offset}`) computes
`coeff * U[0,1) + offset`, matching `iree_corpus.py`.

A `file` (native) or `File` (upstream) input copies an existing `.npy`
relative to the case directory.

## Validation (CEL)

There are two complementary mechanisms.

**1. CEL property predicates** (`validate = [...]`). Each entry is a CEL
expression that must evaluate to `true`. Facts about output 0 are exposed at
the top level; every output is also available via the `outputs` list. All
`[params]` are in scope.

| Fact            | Meaning                                        |
| --------------- | ---------------------------------------------- |
| `count`         | number of elements                             |
| `shape`         | list of dimensions                             |
| `sum` / `mean`  | sum / mean of all elements                     |
| `min` / `max`   | min / max element                              |
| `all_finite`    | true if no NaN/Inf                             |

When an expected tensor is available (see below), these extra facts appear:
`max_abs_diff`, `max_rel_diff`, `mismatches`, `allclose`.

```toml
validate = [
  "all_finite",
  "min >= 0.0",                 # e.g. a ReLU output
  "outputs[0].max <= 1.0",
  "max_abs_diff < 1e-4",        # requires an expected tensor
]
```

**2. Numeric checks against an expected tensor** (`checks = [...]`, or upstream
`expected_output` + `checks`). An expected tensor comes from a `file` output or
a `formula` op (e.g. `matmul` of two inputs). Supported checks:

* `allclose` — `|obs - exp| <= atol + rtol*|exp|` everywhere.
* `sparse_allclose` — allclose at the listed `indices` only.
* `corner_allclose` — allclose on a corner sub-box (`corner`,`shape`).

If a case has an expected tensor but no explicit checks/predicates, a default
`allclose` per output is applied.

## Running

```sh
# All built-in scenarios (unavailable ones skip), all discovered cases:
mirage corpus run --root ./corpus

# A subset of cases under specific scenarios, writing reports:
mirage corpus run --root ./corpus \
  --scenario rocjitsu --scenario native \
  --case matmul_f32 --case relu_f32 \
  --out-dir ./out            # writes results.csv + results.json

# Compile only (skip running/validation):
mirage corpus run --root ./corpus --compile-only

# Run the IREE tools directly, without the mirage run-wrapper:
mirage corpus run --root ./corpus --scenario native --no-wrapper
```

Useful flags: `--config <file>` (repeatable), `--artifact-dir <dir>` (modules,
inputs, outputs, logs; default `.corpus-artifacts`), `--iree-compile` /
`--iree-run-module` (override tool names), `--mirage-bin` (wrapper binary;
defaults to the running `mirage`). `mirage corpus run` exits non-zero if any
case is a real failure (skips and xfails don't count).

`mirage corpus bench` runs the same pipeline and prints per-scenario wall-clock
timings instead of pass/fail labels.

### Output

```text
CASE                             rocjitsu     native
matmul_f32                       pass         pass
relu_f32                         pass         pass

2 passed, 0 failed, 0 skipped
```

`--out-dir` writes `results.csv`
(`config,scenario,case,status,elapsed_s,returncode`) and `results.json`
(the full report). `--json` prints the JSON report to stdout.

## Dashboard

Open the dashboard (`mirage webui`) and visit **Corpus**:

1. Enter a corpus root and click **Discover cases**.
2. Pick scenarios and cases (checkboxes).
3. **Run** — the results matrix shows pass/fail (or timings) per
   case × scenario, with failure messages on hover.

## REST API

| Method | Path                          | Body / Response                         |
| ------ | ----------------------------- | --------------------------------------- |
| GET    | `/api/corpus/scenarios`       | `[CorpusScenario]`                      |
| GET    | `/api/corpus/cases?root=DIR`  | `[Case]`                                |
| POST   | `/api/corpus/run`             | `RunRequest` → `RunReport`              |

`RunRequest`: `{ root, configs?, cases?, scenarios?, compile_only?, no_wrapper?, artifact_dir? }`.

## ctest

`MirageCorpusNative.cmake` registers two hermetic ctests around the demo
fixture (on by default): `corpus_native` (full compile→run→validate, needs
`python3` for the stub runner; skips with code 77 otherwise) and
`corpus_native_list` (a fast loader/CLI smoke check). The separate
`RocjitsuCorpus.cmake` bridge (`MIRAGE_RUN_CORPUS`, opt-in) still drives the
upstream corpus runner through `mirage run`.

## Adding a new test (contributor guide)

1. Drop a `<name>.mlir` (or several) and a `<name>.toml` next to it under your
   corpus root.
2. Give the case a unique `name`, list its `sources`, and set `function`.
3. Generate inputs with a CEL `formula` (or reference a `.npy` file).
4. Validate with CEL `validate` predicates and/or numeric `checks`.
5. `mirage corpus run --root <dir> --case <name> --scenario native` (add
   `--no-wrapper` while iterating locally).
6. Once green, run it under `rocjitsu`/`hotswap` as available.

[corpus]: https://github.com/rocm/rocjitsu-test-corpus
[cel]: https://github.com/google/cel-spec
