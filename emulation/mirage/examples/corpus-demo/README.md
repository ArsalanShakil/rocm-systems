# corpus demo

A tiny, self-contained `mirage corpus` example. It runs a single ReLU kernel
case through the full **compile → run → validate** pipeline using *stub* IREE
tools, so it works with **no real IREE install and no GPU**.

```sh
# from anywhere, using the mirage on your PATH:
./demo.sh

# or point at a freshly built binary:
MIRAGE=../../target/debug/mirage ./demo.sh
```

## What's here

| File                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `relu.toml`           | The corpus case: CEL-generated 4×4 input, CEL output checks.   |
| `relu.mlir`           | Placeholder MLIR source (not compiled by the stub tool).       |
| `tools/iree-compile`  | Stub compiler — writes a dummy `.vmfb`.                        |
| `tools/iree-run-module` | Stub runner — computes ReLU on the input `.npy` (python3).   |
| `demo.sh`             | Drives `scenarios` / `list` / `show` / `run`.                 |

`demo.sh` prepends `tools/` to `PATH` so the stubs stand in for a real IREE.
The case generates its input from a CEL formula and validates the output with
CEL predicates (`all_finite`, `min >= 0.0`, `count == 16`, `shape == [4, 4]`),
so the run is fully deterministic.

The stub `iree-run-module` needs `python3`. This same fixture backs the
`corpus_native` ctest and the `tests/corpus_native_e2e.rs` end-to-end test.

See [`docs/corpus.md`](../../docs/corpus.md) for the full reference.
