# Simpler L4 Direct MPI L3 替换与验证设计

## 0. 四项闭环与实施顺序

最终替换需要完成四项闭环，并按可独立验证的阶段合入：

1. **MPI 进程闭环**：一次 `mpirun` 在两机各拉起一个真实 L3 rank，完成拓扑校验、
   L2 子树初始化、READY、失败传播和有界退出。
2. **L4 控制面闭环**：L4 的 OPEN_SESSION、HELLO、CONTROL、TASK、COMPLETION、
   HEALTH、SHUTDOWN 经本机 UDS 和跨机 MPI 到达真实 L3；不调用默认 Remote L3 TCP
   daemon，也不创建 TCP session runner。
3. **Fabric handle 闭环**：源 L2 export Fabric window，handle 经 L3/MPI 交换，目标
   L2 import/map/release；所有异常分支可回收。
4. **Fabric 数据访问闭环**：目标 L2 将 RemoteTensor 解析为 importer-local GVA，使用
   PTO `TLOAD/TSTORE` 完成真实跨机读写和 golden 校验。

每阶段必须同时满足以下合入门槛：

- 修改直接进入 Simpler L4/L3 runtime，不交付旁路通信冒烟程序。
- 新能力只有显式配置才启用；默认 `control_transport="socket"` 的
  L4 -> L3 -> L2 路径保持可用。
- 阶段内有单元测试和端到端证据；未完成的下一阶段能力不作为本阶段结论。
- 失败或关闭后不留下 rank、线程、UDS、shm、NPU 资源或 Fabric window。

当前代码已实现并可验证第 1、2 项。第 3、4 项仍是后续阶段；第一阶段通过只证明
MPI 可以承载 Simpler 跨机控制帧并调度真实 NPU，不证明 Fabric 数据面已经完成。

## 1. 当前结论

新路径不再使用独立中转 daemon、Python proxy 或远端 TCP session runner。
`mpirun` 直接拥有两台机器上的 L3 进程：

```text
机器 A                                                        机器 B

+----------------------+
| L4 master            |
| Worker(level=4)      |
+----------+-----------+
           | AF_UNIX command/health
           v
+----------------------+       MPI P2P       +----------------------+
| rank 0 / real L3     | <=================> | rank 1 / real L3     |
| Worker(level=3)      |                     | Worker(level=3)      |
| local MPI gateway    |                     | MPI route + executor |
+----------+-----------+                     +----------+-----------+
           | existing shm/mailbox                       | existing shm/mailbox
           v                                            v
+----------------------+                     +----------------------+
| L2 ChipWorker 0 / 1 |                     | L2 ChipWorker 0 / 1 |
| NPU device 0 / 1    |                     | NPU device 0 / 1    |
+----------------------+                     +----------------------+
```

路径边界明确如下：

| 段 | MPI L3 路径 | 默认路径 |
|---|---|---|
| L4 -> 本机 rank 0 | AF_UNIX UDS | TCP bootstrap + TCP session |
| rank 0 -> rank 0 L3 | 同进程本地路由 | TCP session |
| rank 0 -> rank 1 L3 | MPI P2P opaque envelope | TCP session |
| L3 -> L2 | 原有 shm/mailbox | 原有 shm/mailbox |

`mpirun` 自身可以使用 SSH/TCP 启动远端进程，MPI 实现也可以选择 TCP、UCX 或 RDMA。
这属于 MPI launcher/transport，不是 Simpler 默认 `RemoteL3SocketTransport` 链路。

## 2. 第一阶段具体逻辑

### 2.1 进程和初始化顺序

机器 A 的 launcher 只执行一次 `mpirun -np 2`。每个 rank 运行
`python -m simpler.mpi_l3_worker`，并直接构造一个 `Worker(level=3)`。

每个 L3 必须遵守 Simpler 的 fork/thread 契约：

1. L3 创建 L2 mailbox 和 native Worker。
2. L3 fork 本机两个 L2 ChipWorker，并等待 L2 INIT_READY。
3. 最后一个本地 fork 完成后，通过 `_post_fork_pre_threads` 初始化 MPI。
4. MPI 初始化完成后才启动 L3 Scheduler/WorkerThread。
5. 两个 rank 执行 barrier，rank 0 随后发布 L4 bootstrap UDS。

这样 L2 子进程不会继承已初始化的 MPI runtime 或 MPI 线程。

### 2.2 L4 -> L3 session 建立

L4 显式使用：

```python
RemoteWorkerSpec(
    endpoint="mpi://rank/1",
    platform="a2a3",
    control_transport="mpi_l3",
    gateway_endpoint="/tmp/.../l4-gateway.sock",
    mpi_rank=1,
    device_ids=(0, 1),
)
```

`Worker.init()` 在本机所有 fork 完成后连接 rank 0 bootstrap UDS，发送长度前缀 JSON：

```text
OPEN_SESSION(target_rank, manifest, startup_remaining_s)
```

rank 0 校验 rank 和 session，封装为 MPI envelope。目标 rank 校验 manifest 必须与
预启动 L3 的 `platform/runtime/device_ids/num_sub_workers/worker_id` 完全一致，然后为
该 session 创建两对本机 `socketpair`：

- command：连接 SLR3 command executor 和 MPI router。
- health：连接 health producer 和 MPI router。

executor 直接调用现有 `remote_l3_session._run_command_loop()`，但使用已经初始化的
本 rank L3 Worker，不再 fork/exec 另一个 L3。目标返回 READY 后，rank 0 创建该
session 的 command/health UDS，L4 的 `RemoteL3UnixTransport` 完成 HELLO attach。

### 2.3 运行期帧路由

SLR3 帧保持字节透明，不在 MPI 层重新解释 task payload：

```text
L4 RemoteL3Endpoint
  -> RemoteL3UnixTransport
  -> rank 0 source session
  -> FRAME_L4_TO_L3 envelope
  -> local dispatch 或 MPI P2P
  -> target rank command socketpair
  -> existing _run_command_loop
  -> L3 Orchestrator
```

返回路径使用显式 `FRAME_L3_TO_L4`，不能通过 source rank 推断方向，因为 rank 0 的
本机 worker session 两端都属于 rank 0。每个 envelope 包含：

- magic/version/message type；
- source/target rank；
- session id、lane、sequence；
- 原始 SLR3 frame bytes。

MPI 调用只发生在各 rank 主线程。UDS reader、command executor 和 health producer
线程只向 rank 主线程队列投递 envelope，因此只要求 `MPI_THREAD_FUNNELED`。

### 2.4 L3 -> L2 保持不变

目标 L3 收到 TASK/CONTROL 后继续走原始代码：

```text
_run_command_loop
  -> inner Worker(level=3)
  -> Orchestrator / Scheduler
  -> shm mailbox
  -> L2 ChipWorker
  -> NPU runtime
```

MPI 不进入 L2、不替换 mailbox、不改变 ChipCallable ABI。当前真实 NPU case 在每台
机器使用 device 0、1，并由 L3 的 `submit_next_level_group(..., workers=[0, 1])`
调度两个 L2。

### 2.5 关闭和故障传播

- L4 close 通过已有 command lane 发 SLR3 SHUTDOWN。
- command EOF 或异常触发 CLOSE_SESSION，关闭 source/target UDS、socketpair 和线程。
- launcher 通过 rank 0 bootstrap UDS 发 STOP_WORLD；rank 0 广播 MPI SHUTDOWN。
- 任一 rank 未处理异常调用 MPI abort，避免另一个 rank 永久等待。
- launcher 对 rank world、L4 case 和 shutdown 都使用同一个配置超时上限。

当前第一阶段限制每个 rank 同时只有一个 L4 session，MPI world 固定为两个 rank。

## 3. 默认 TCP 路径隔离

默认配置仍为：

```python
RemoteWorkerSpec(
    endpoint="120.9.10.35:19073",
    platform="a2a3",
    control_transport="socket",
)
```

它继续执行：

```text
L4
  -> TCP bootstrap -> simpler.remote_l3_worker
  -> TCP command/health -> remote L3 session runner
  -> L3 mailbox -> L2
```

隔离约束：

- `socket` 不接受 `gateway_endpoint` 或 `mpi_rank`。
- `mpi_l3` 使用逻辑 `mpi://rank/N`，不接受 TCP session listener 字段。
- MPI 构建产物是独立共享库，不链接到默认 Simpler runtime。
- MPI launcher 不检查端口 19073，也不启动 `simpler.remote_l3_worker`。
- `add_remote_l3_socket()` 和 `RemoteL3SocketTransport` 的实现未被替换。

## 4. 代码归属

```text
python/simpler/worker.py
  RemoteWorkerSpec 的 socket/mpi_l3 显式分支
  MPI bootstrap UDS attach 和 RemoteL3UnixTransport 注册

python/simpler/mpi_l3_gateway.py
  UDS bootstrap、session、MPI envelope、local/remote route、L3 executor

python/simpler/mpi_l3_worker.py
  mpirun rank 入口、L3/L2 初始化、MPI ctypes bridge 生命周期

src/common/hierarchical/remote_endpoint.{h,cpp}
  RemoteL3SocketTransport（默认 TCP）
  RemoteL3UnixTransport（MPI 路径的本机 L4 接入）

src/common/hierarchical/mpi_l3_transport/
  可选 MPI C++ bridge，仅暴露 init/send/probe/recv/barrier/finalize/abort

tools/mpi_l3/
  构建、两机启动、生产拓扑、真实 L4/L3/L2/NPU 验证
```

旧中转进程、旧协议目录、旧 CLI、旧 launcher 和旧测试已删除；新代码不 import、
exec 或链接这些实现。

## 5. 第一阶段验证

### 5.1 静态和单元验证

```bash
python -m py_compile \
  python/simpler/mpi_l3_gateway.py \
  python/simpler/mpi_l3_worker.py \
  tools/mpi_l3/launch.py \
  tools/mpi_l3/npu_e2e_case.py

pytest -q tests/ut/py/test_mpi_l3.py tests/ut/py/test_mpi_l3_spec.py
```

单元测试必须证明：逻辑 rank 映射、local route 不做 MPI self-send、remote route 进入
MPI、MPI 路径不调用 TCP bootstrap、UDS bootstrap 不携带 daemon host/port，以及
默认 socket 配置不变。

### 5.2 两机真实 NPU 验证

仓库生产拓扑已经配置机器 A `120.9.10.37`、机器 B `120.9.10.35`，两边均使用
device 0、1，并显式配置 `worker_id == rank` 为 0/1。在机器 A 的一个终端运行：

```bash
source .venv/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

bash tools/mpi_l3/build.sh
bash tools/mpi_l3/run_2host_npu.sh tools/mpi_l3/topology.2host-npu.json
```

无需另外开启 Remote L3 TCP 监听终端。预期关键证据：

```text
MPI_INIT_AFTER_L2_FORK             x2
MPI_L3_WORLD_READY                 x2
L4_MPI_L3_GATEWAY_READY            x1
MPI_L3_SESSION_READY               x2
MPI_L3_ROUTE_LOCAL                 >=1
MPI_L3_SEND / MPI_L3_RECV          >=1
status=PASS, control_transport=mpi_l3
command frame sequence/hash comparison PASS
direct MPI L3 validation PASS
```

launcher 按 `(session_id, worker_id, frame_type, sequence, sha256)` 比较
L4_TO_MPI/MPI_TO_L3 和 L3_TO_MPI/MPI_TO_L4，确保 TASK/COMPLETION 没有丢帧、重排或
内容变化。case 同时回读两机四组 NPU 输出并进行 golden 校验。

默认 TCP 回归应使用仓库现有 Remote L3 测试入口独立执行；它不应由 MPI launcher
隐式启动。两条路径的验证结果可以比较，但运行时互不依赖。

## 6. 后续阶段

### 阶段二：生命周期和故障门禁

在不改变 wire payload 的前提下补齐 rank crash、MPI send/recv error、L4 timeout、
重复/迟到 frame、健康通道中断和 launcher signal 的双机注入测试。验证默认 TCP
回归、MPI 正常 case 和每个故障 case 均能有界退出且无资源残留。

### 阶段三：Fabric handle 闭环

在现有 RemoteBuffer CONTROL protocol 增加结构化 Fabric export/import 描述符。
owner L2 负责 window 生命周期，目标 L2 返回 import id；L3/MPI 只做结构化转发和
身份校验。先验证 export -> MPI exchange -> import -> release，不执行 PTO 数据访问。

### 阶段四：Fabric 数据访问闭环

将 RemoteTensor 的 owner 身份解析为 importer-local GVA，执行跨机 `TLOAD/TSTORE`
kernel，校验数据、generation、防止 stale handle，并覆盖正常关闭、任务失败和 rank
失败下的 release/revoke。完成后才能声称 MPI 路径替代了控制面并支持 Fabric 数据面。

## 7. 第一阶段非目标

- 不要求 MPI 底层禁用 TCP。
- 不把 L4 master 放进 MPI world。
- 不让 MPI 进入 L2 进程。
- 不修改默认 TCP daemon/session 协议。
- 不把 CONTROL frame 的 RemoteBuffer copy 宣称为 Fabric handle 或零拷贝数据面。
