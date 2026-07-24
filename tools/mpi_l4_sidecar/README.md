# L4 MPI Sidecar Validation

These files only build, launch, and validate the MPI transport implemented in
`src/common/hierarchical/mpi_sidecar/`. The TASK and control semantics remain
in Simpler's production `RemoteL3Endpoint` and remote L3 session.

## Build

Both hosts need the same MPI implementation/ABI and an MPI C++ compiler:

```bash
bash tools/mpi_l4_sidecar/build.sh
```

The default Simpler build and import do not require MPI.

## One Host, Two Ranks

Create a local config from `topology.example.json`. `mpi_command` is the full
launcher prefix before the sidecar executable, so MPI-specific flags remain
explicit data rather than shell evaluation.

```bash
cp tools/mpi_l4_sidecar/topology.example.json \
  tools/mpi_l4_sidecar/topology.local.json
bash tools/mpi_l4_sidecar/run_1host_sim.sh \
  tools/mpi_l4_sidecar/topology.local.json
```

The launcher starts the existing remote daemon and two local proxies, waits for
both proxy sockets, then starts MPI. It runs the same callable twice:

```text
legacy_socket_sim: L4 -> TCP -> remote L3 -> L2 sim
mpi_sidecar_sim:   L4 -> UDS -> MPI P2P -> proxy -> remote L3 -> L2 sim
```

Both runs dynamically register the same inner L3 callable and execute it on a
real L2 sim child. The launcher compares the structured result, requests
rank-0 world shutdown, waits for `MPI_Finalize`, and rejects Unix socket
residue. Proxy logs include rank, lane, SLR3 frame type, sequence, and SHA-256
for comparing TASK/COMPLETION forwarding. Target cleanup also records
`REMOTE_L3_SESSION_CLOSED` with `runner_exited=true`; failure to observe runner
exit within the cleanup bound fails the normal CLOSE path. The launcher compares
the command-lane sequence/hash multisets at both MPI boundaries and requires a
TASK and COMPLETION pair.

## Two Hosts

The first host is rank 0 and runs the L4 process locally. The second host is
rank 1 and owns remote worker 0. Passwordless SSH and a shared MPI placement
configuration are prerequisites.

```bash
cp tools/mpi_l4_sidecar/topology.2host.example.json \
  tools/mpi_l4_sidecar/topology.2host.json
bash tools/mpi_l4_sidecar/run_2host_sim.sh \
  tools/mpi_l4_sidecar/topology.2host.json
```

`work_dir` and `sidecar_binary` must resolve to the same absolute paths on both
hosts. `mpi_command` must place exactly rank 0 on the local host and rank 1 on
the configured SSH host. The rank-1 daemon endpoint must be reachable both
from rank 0 for the legacy comparison and from rank 1 itself for sidecar
bootstrap.

Passing requires all of these records and zero exit status:

```text
MPI world READY
L4 bootstrap READY
legacy socket L4 -> L3 -> L2 PASS
MPI sidecar L4 -> L3 -> L2 PASS
legacy_vs_mpi PASS
mpi_frame_sequence_hash_compare PASS
no Unix socket residue
```

The launcher only terminates processes it started for its unique job directory;
it does not use global process names or `killall`.
