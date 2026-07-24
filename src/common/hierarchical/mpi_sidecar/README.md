# MPI L4 Sidecar

This executable is the optional MPI control-plane relay for Simpler's real
`RemoteL3Endpoint`. It is not linked into `_task_interface` and is not part of
the default CMake build. Build it only on hosts with an MPI C++ toolchain:

```bash
bash tools/mpi_l4_sidecar/build.sh
```

The launcher starts one `simpler.remote_l3_sidecar_proxy` per MPI rank before
starting this executable with `mpirun`. The sidecar then:

1. connects the rank-local proxy before `MPI_Init_thread`;
2. initializes MPI with `MPI_THREAD_FUNNELED`;
3. verifies the shared topology ID and worker map with `MPI_Allgather`;
4. publishes `MPI world READY` to the proxy;
5. forwards versioned envelopes with MPI P2P while leaving SLR3 payloads
   unchanged;
6. exits through a rank-0 bounded world shutdown.

Only the sidecar main thread calls MPI. The proxy owns JSON parsing, Unix
sockets, remote-daemon bootstrap, and runner command/health TCP sockets.
Consequently L4/L3 Worker processes never initialize MPI and preserve their
existing fork-before-thread lifecycle.

The envelope is little-endian and has a fixed 44-byte header:

```text
magic/version/type/source_rank/target_rank/session_id/lane/payload_bytes/sequence
```

Payloads are bounded at 16 MiB. `FRAME` payloads are canonical SLR3 bytes and
must match the envelope's session and sequence. Command and health are separate
logical lanes.

This first-stage implementation covers bootstrap, HELLO, callable control,
TASK, COMPLETION, HEALTH, SHUTDOWN, and cleanup for the sim L4 -> L3 -> L2
path. Fabric handle exchange and Fabric data access remain later stages.
