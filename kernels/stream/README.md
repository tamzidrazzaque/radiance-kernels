# STREAM on Radiance

The [workload result summary](../README.md) combines these cycles
with the Spatter measurements.

The four independent STREAM operations use float32 arrays: Copy `C=A`, Scale
`B=2*C`, Add `C=A+B`, and Triad `A=B+2*C`. They build through `../common.mk`
and run on the same Radiance SoC as `kernels/spatter`. The generated
inputs are exact small integers in float32, so the expected output digest does
not depend on rounding differences. One operation runs per ELF; a complete
four-stage STREAM sequence would use a different data dependency and timing
contract.

Build each case with `LLVM_MUON`, `RISCV_TOOLCHAIN_PATH`,
`RISCV64_TOOLCHAIN_PATH`, `RISCV`, and `MU_STACK_WORD_STRIDE` set for the same
toolchain as the runtime library. The installed older compiler requires
`MU_STACK_WORD_STRIDE=1`. On a fresh checkout, plain `make` creates a
256-element Copy smoke input; later calls reuse the last generated case.

```sh
python3 run.py --kind copy --elements 65536 --out runs/copy-65536
python3 run.py --kind scale --elements 65536 --out runs/scale-65536
python3 run.py --kind add --elements 65536 --out runs/add-65536
python3 run.py --kind triad --elements 65536 --out runs/triad-65536
```

Build cases sequentially because they share `generated/`. Each output has a
manifest, build log, and fused ELF. The RV64 host checks a complete digest for
up to 1024 elements and 64 samples for larger arrays; Cyclotron can check the
complete output. `output_elements` in the manifest is the number of 64-bit
pairs passed to Spatter's reusable Cyclotron checker; `stream_elements` is the
actual number of float32 values.

The default is the one-cluster/two-core topology. Pass `--muon-clusters 2`
only for a Radiance TwoCluster target (two clusters with two Muon cores each).
The selected cluster, core, warp and thread counts are recorded in the
manifest and emitted into `generated/config.h`, so changing the option rebuilds
the device object. The compiled cluster count must match the target: a
one-cluster binary on a two-cluster target duplicates work across clusters and
can still pass a deterministic output check while reporting invalid timing.

Set `MU_ADDR_HASH=1` for the newer hashed `RadianceHBMConfig` when loading
DRAM directly with `+loadmem` or FireSim LoadMem; the hash parameters must
match the RTL. Use `MU_ADDR_HASH=0` for the earlier unhashed image or a TSI
loader. The Makefile rebuilds the fused ELF when the hash setting changes.
The manifest records the effective size, unit, slice count, GPU host base and
required load path. The results below remain tied to their recorded hardware
image and ELF.

For a complete host check on a larger array, add `--full-host-check`. Add
`--host-timing` to print `HOST_RELEASE_TO_DONE_CYCLES` in the guest UART. This
counts RV64 host cycles from just before Muon reset release until the
END mailbox from every selected Muon core is observed. Every kernel thread
first reaches a block barrier; one warp then executes `mu_fence` on each core
before its lane 0 posts END. The interval therefore ends after the selected
cores' L0d fences complete and excludes the later UART report and digest
readback. The UART also reports `kernel_cycles` from the earliest per-core
START to the latest END in Muon `mcycle` units. Targets must correctly decode
`lw.shared`/`sw.shared` for the mailboxes and implement the per-core fence;
raw `GPU_ALL_FINISHED` is not a substitute. The host poisons the complete
NOBITS output with quiet NaNs before the timestamp, so skipped or partial
writes cannot pass by inheriting a previous result. Timed and untimed builds
retain byte-identical embedded RV32 images.

Use `../spatter/tools/spatter-verilator-run.py` and
`../spatter/tools/spatter-cyclotron-run.py` with `--source-root runs` and a
separate output root to run an ELF. The shared validator and summarizer produce
the same cycle and logical-payload tables as the Spatter kernels. Logical
payload counts two float32 transfers per element for Copy/Scale and three for
Add/Triad; it excludes cache-line amplification. No GPU clock or calibrated
HBM bandwidth is assumed.

See [evaluation/README.md](evaluation/README.md) for the measured
one-million-element timing-model cycle and memory counters.
