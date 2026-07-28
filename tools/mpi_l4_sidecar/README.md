# L4 MPI Sidecar Validation

These files only build, launch, and validate the MPI transport implemented in
`src/common/hierarchical/mpi_sidecar/`. The TASK and control semantics remain
in Simpler's production `RemoteL3Endpoint` and remote L3 session.

The no-device sim runs are fast regression gates. The phase-1 hardware gate is
the two-machine/two-rank case below: it reuses the real vector L3-group workload from
`l4test/tools/remote_l4_npu` and compares the existing TCP path with the MPI
sidecar path.

## Build

All participating hosts need the same MPI implementation/ABI and an MPI C++ compiler:

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

## Two Hosts With The Master On Machine A

This mode matches `l4test/tools/remote_l4_npu`: there are exactly two machines.
Each machine runs one remote L3 daemon and one MPI sidecar rank. Machine A also
runs the existing L4 master process; the master is not started by `mpirun` and
does not consume an MPI rank:

```text
machine A / rank 0: L4 master + bootstrap/source + worker 0 -> NPU 0,1
machine B / rank 1:                         worker 1 -> NPU 0,1
worker_map: 0:0;1:1
```

Protocol v2 carries an explicit frame direction so the rank 0 proxy can route
both master-to-local-L3 and local-L3-to-master frames without inferring the
direction from equal source/target ranks.

Start the normal remote daemon on both NPU hosts from a checkout containing the
same code. Source the local CANN environment first; the exact setup remains an
installation concern rather than launcher configuration:

```bash
source .venv/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
python -m simpler.remote_l3_worker --host 0.0.0.0 --port 19073
```

The tracked `topology.2host-npu.json` contains the production placement for
`120.9.10.37` and `120.9.10.35`. Use
`topology.2host-npu.example.json` when validating another machine pair.

On the production parent, build the sidecar and run the tracked topology:

```bash
source .venv/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
bash tools/mpi_l4_sidecar/build.sh
bash tools/mpi_l4_sidecar/run_2host_npu.sh \
  tools/mpi_l4_sidecar/topology.2host-npu.json
```

The launcher leaves the two pre-existing daemons running. One `mpirun` starts
both sidecar ranks; each rank starts and owns its local Python proxy. There is
no separate Simpler SSH command or `hosts[].ssh` topology field. On bare hosts,
OpenMPI/MPICH may still use passwordless SSH internally to launch the remote
MPI rank, just as `mpirun-test/tools/a3_hccl_smoke` does.

As in `mpirun-test`, the launcher creates a per-run hostfile. MPICH uses
`-f <hostfile> -ppn 1 -np 2`; OpenMPI uses a two-entry, one-slot-per-host
hostfile with `-np 2`. The legacy full `mpi_command` array remains accepted,
but `mpi` plus `mpi_host` is the preferred NPU configuration.

The launcher first executes the real NPU task through the default TCP control
path, then executes the same task through UDS/MPI P2P. Both runs dynamically install the same `ChipCallable`,
allocate/copy/free the same remote buffers, submit the same two-device L3 group,
and validate the same golden values. It also compares output SHA-256 records and
aggregates frame sequence/hash records from the combined MPI output. Each rank
waits for its local proxy to exit and rejects a leftover proxy UDS before the
MPI job can report success.

`transport="sim"` in the NPU case remains the remote-buffer transport profile;
it does not select an L2 simulator. `platform="a2a3"` plus the two `device_ids`
causes the inner L3 worker to fork and dispatch to real L2 NPU children. This
phase copies input/output through remote control messages and does not claim
cross-machine NPU Fabric import/export.
