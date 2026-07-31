# Direct MPI L3 Validation

`mpirun` 在两台机器各启动一个真实 `Worker(level=3)`。L4 master 在机器 A
单独运行，通过本机 UDS 接入 rank 0；跨 rank 的 SLR3 控制帧由 MPI P2P 承载。
两个 L3 都继续使用原有 shm/mailbox 调度本机两个 L2 ChipWorker。

```text
L4 master
  -> machine A UDS
  -> MPI rank 0 / L3
       |-- worker 0: local in-process route
       `-- worker 1: MPI P2P -> rank 1 / L3

each L3 -> existing shm/mailbox -> L2 0/1 -> NPU
```

默认 `RemoteWorkerSpec.control_transport="socket"` 和
`simpler.remote_l3_worker` 不受此入口影响。MPI 路径必须显式选择
`control_transport="mpi_l3"`，且不启动默认 TCP daemon。

## Prerequisites

- 两机存在相同绝对路径的 checkout、Python virtualenv 和构建产物。
- 两机使用相同 MPI 实现和 ABI，master 的 `mpirun` 可以拉起远端 rank。
- 每台机器的 JSON 所列 NPU device 可用。
- 第一阶段拓扑固定 `worker_id == rank`，即 0/1 一一对应。
- CANN 环境可被 MPI launcher 继承。

MPI 实现可以用 SSH/TCP 启动和传输 rank；这不等于使用 Simpler 默认 Remote L3
TCP 控制链路。

## Build And Run

在机器 A 的一个终端执行：

```bash
source .venv/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

bash tools/mpi_l3/build.sh
bash tools/mpi_l3/run_2host_npu.sh tools/mpi_l3/topology.2host-npu.json
```

脚本生成 MPI hostfile，启动两个 L3 rank，等待 rank 0 UDS gateway，运行真实
L4/NPU case，关闭 MPI world，并检查 UDS 残留。无需另外启动
`simpler.remote_l3_worker`，也无需三个终端。

成功时至少看到：

```text
{"event": "MPI_INIT_AFTER_L2_FORK", ...}        # 两个 rank
{"event": "MPI_L3_WORLD_READY", ...}            # 两个 rank
{"event": "L4_MPI_L3_GATEWAY_READY", ...}       # rank 0
{"event": "MPI_L3_SESSION_READY", ...}          # 两个 L3
{"event": "MPI_L3_SEND", "target_rank": 1, ...}
{"event": "MPI_L3_RECV", "source_rank": 1, ...}
{"status": "PASS", "control_transport": "mpi_l3", ...}
[mpi-l3] command frame sequence/hash comparison PASS (...)
[mpi-l3] direct MPI L3 validation PASS; logs: /tmp/simpler-mpi-l3-...
```

`transport="sim"` 仍表示第一阶段 RemoteBuffer profile，不代表 L2 simulator；
`platform="a2a3"` 和 `device_ids` 决定真实 NPU 执行。第一阶段输入输出仍使用
CONTROL frame 搬运，不包含跨机 Fabric handle import/export。

完整边界、阶段和验证标准见
[`mpi_sidecar_replacement_validation_design.md`](../../mpi_sidecar_replacement_validation_design.md)。
