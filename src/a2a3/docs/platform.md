# Platform Backends (a2a3)

Two platform backends under `src/a2a3/platform/`, providing different execution environments for the same runtime code.

## Comparison

| Feature | onboard | sim |
| ------- | ------- | --- |
| Execution | Real Ascend hardware | Thread-based host simulation |
| Requirements | CANN toolkit, `ccec`, aarch64 cross-compiler | gcc/g++ only |
| AICore compilation | `ccec` (Bisheng CCE compiler) | g++ with `-D__CPU_SIM` |
| AICPU compilation | aarch64-target-linux-gnu-g++ | Host g++ |
| Host compilation | Host g++ | Host g++ |
| Device memory | Real GM/L1/L2 via Ascend driver | `malloc`-backed simulation |
| Use case | Production, hardware validation | Development, debugging, CI |

## onboard

Real hardware backend. Requires:

- `ASCEND_HOME_PATH` environment variable pointing to the Ascend toolkit
- `ccec` compiler for AICore kernels
- aarch64 cross-compiler for AICPU code

Key directories:

- `src/a2a3/platform/onboard/host/` — Host runtime library (device_runner, memory_allocator)
- `src/a2a3/platform/onboard/aicpu/` — AICPU kernel entry and platform registers
- `src/a2a3/platform/onboard/aicore/` — AICore kernel build (ccec + ld.lld)

### SDMA Prefetch

The onboard `tensormap_and_ringbuffer` runtime can optionally create STARS
SDMA prefetch channels on the host and let AICPU schedulers issue CMO PREFETCH
SQEs for the next task in a dispatch batch.

| Environment variable | Default | Purpose |
| -------------------- | ------- | ------- |
| `PTO_SDMA_PREFETCH_MODE` | `sdma` unless legacy disable env is set | `baseline`/`0` and `twoslot`/`1` disable real prefetch; `sdma`/`2` enables it; `sdma_fake`/`3` keeps the mode explicit without issuing real SDMA. |
| `PTO_SDMA_PREFETCH_MIN_BYTES` | `262144` | Minimum readable task bytes before the scheduler considers prefetch. |
| `PTO_SDMA_PREFETCH_SUPPRESS_WINDOW` | `2` | Per-channel eligible-attempt suppression after a successful issue. |
| `PTO_SDMA_PREFETCH_CHANNELS` | worker count | Caps host-created STARS channels to `min(env, worker_count)`. |
| `PTO_SDMA_PREFETCH_DEBUG` | off | Emits prefetch control-path and issue counters in device logs. |

Simulation backends provide no-op stubs for the same AICPU symbols.

## sim

Thread-based simulation. No hardware or SDK required. Each AICore/AICPU "device" runs as a host thread.

Key directories:

- `src/a2a3/platform/sim/host/` — Simulated device runner and memory
- `src/a2a3/platform/sim/aicpu/` — Simulated AICPU executor
- `src/a2a3/platform/sim/aicore/` — Simulated AICore executor

## Shared Interface

Platform-agnostic headers live in `src/a2a3/platform/include/`, split by target:

- `host/` — Host-side platform API
- `aicpu/` — AICPU platform API (registers, timing)
- `aicore/` — AICore platform API
- `common/` — Shared types and utilities (unified_log, tensor, common.h)

Shared source implementations in `src/a2a3/platform/src/`.
