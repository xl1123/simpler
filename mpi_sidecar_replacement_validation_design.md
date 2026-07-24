# Simpler L4 MPI Sidecar 集成与验证方案

## 1. 设计结论

本方案直接修改 Simpler 的 L4 跨机实现，不以
`mpirun-test/tools/` 下的独立冒烟程序作为交付结果。第一阶段完成后，真实
`Worker(level=4)` 必须能够显式选择 MPI sidecar，经过远端 L3 调度到 L2，并收到
任务完成结果；仅有 `mpirun` rank 互发消息不能算通过。

“不影响原有生产路径”的含义是：代码会修改，但修改是增量且默认关闭。原有
L4 -> TCP Remote L3 -> 本机 mailbox L2 路径保持默认、接口兼容和功能可用；不是指
不修改 `simpler`。

```text
原有默认路径

L4 Worker
  -> RemoteL3SocketTransport
  -> 跨机 TCP command + health
  -> remote L3 session
  -> 原有 L3 Scheduler / shm mailbox
  -> L2 ChipWorker

新增显式路径

L4 Worker
  -> RemoteL3SidecarTransport
  -> 本机 UDS
  -> MPI sidecar rank 0
  -> 跨机 MPI P2P
  -> 远端 MPI sidecar rank N
  -> 本机 proxy / remote L3 session
  -> 原有 L3 Scheduler / shm mailbox
  -> L2 ChipWorker
```

MPI 只进入独立 sidecar 进程，不进入会 fork L2 的 L4/L3 Worker 进程。L4/L3 runtime
通过 UDS 使用 sidecar，因此不要求 `_task_interface` 链接 MPI，也不要求 Worker 使用
`MPI_THREAD_MULTIPLE`。

基线版本：`simpler` `c032e07e`。

## 2. 四项闭环

完整交付需要形成四项闭环：

1. **mpirun 进程闭环**：`mpirun` 在每台参与主机启动一个 MPI sidecar rank，完成
   `worker_id -> mpi_rank -> hostname` 校验、就绪和有界退出。
2. **L4 控制面闭环**：真实 L4 的 bootstrap、HELLO、TASK、CONTROL、COMPLETION、
   HEALTH 和 SHUTDOWN 的跨机段走 MPI P2P。
3. **Fabric handle 闭环**：源 L2 创建/export Fabric window，handle 经
   `L2 -> L3 -> sidecar -> MPI -> sidecar -> L3 -> 目标 L2` 交换，目标 L2 完成
   import/map/release。
4. **Fabric 数据访问闭环**：目标 L2 把逻辑 RemoteTensor 身份解析为 importer-local
   GVA，通过 PTO `TLOAD/TSTORE` 验证真实跨机数据和完整资源回收。

四项能力分四个可合入阶段完成。每个阶段结束时：

- 新增能力有独立测试和端到端结果。
- 原有 socket/sim 路径运行同一类用例并通过。
- 新路径必须显式配置，旧配置语义不变。
- 新路径启动失败不能污染同进程中的旧 socket Worker。
- 失败和关闭后没有 sidecar、runner、UDS、shm 或 Fabric window 残留。

## 3. 兼容性约束

### 3.1 配置和 API

`RemoteWorkerSpec` 增加控制面选择字段：

```python
RemoteWorkerSpec(
    endpoint="10.0.0.8:19073",
    platform="a2a3sim",
    transport="sim",
    control_transport="socket",       # 默认值，保持现有行为
    sidecar_endpoint=None,             # mpi_sidecar 路径必填
    mpi_rank=None,                     # mpi_sidecar 路径必填
)
```

约束如下：

- `control_transport="socket"` 保持当前 `endpoint="host:port"` 解析、numeric host
  限制、bootstrap、command/health socket 和 timeout 行为。
- 只有 `control_transport="mpi_sidecar"` 才解析 `sidecar_endpoint` 和 `mpi_rank`。
- `transport` 继续表示数据/通信 profile，第一、二阶段仍使用 `"sim"`；不能把
  `control_transport` 和 `transport` 混成同一字段。
- 不新增环境变量或编译宏作为选择开关。
- `add_remote_l3_socket()` 和 `RemoteL3SocketTransport` 保留；新增
  `add_remote_l3_sidecar()` 和 `RemoteL3SidecarTransport`。

### 3.2 生命周期

MPI sidecar attach 必须遵守当前 `Worker.init()` 契约：

1. 所有本地 L2/L3 fork 完成后才连接 sidecar。
2. sidecar bootstrap、远端 runner 启动和 HELLO attach 共用当前根启动 deadline。
3. runtime command timeout 与 startup deadline 保持分离。
4. 所有 endpoint 成功后才能原子发布 READY。
5. 任一步失败进入当前统一 rollback，Worker 进入 FAILED，不留下半注册 endpoint。

不得在 `_init_hierarchical()` 的 pre-fork 阶段启动 MPI、proxy 或通信线程。

### 3.3 调度和 L3 -> L2

- Scheduler 继续只看到 `WorkerEndpoint` 和 stable `worker_id`，不感知 MPI rank。
- L4 继续显式通过 `worker=` 选择远端 L3。
- `RemoteL3Endpoint` 继续负责 ordered command lane 和 task/control 语义。
- L3 -> L2 继续使用现有 fork、shm/mailbox、Scheduler 和 ChipWorker ABI。
- 第一阶段的 MPI 用例必须实际包含一个 L2 sim child，不能只在远端 Python dispatcher
  返回结果。

## 4. 第一阶段：直接接入 L4 的最小 MPI 控制闭环

### 4.1 阶段目标

第一阶段同时完成最小的 mpirun 进程闭环和 L4 控制闭环：

```text
mpirun 启动至少两个 sidecar rank
  -> rank/host/worker 拓扑 READY
  -> L4 Worker.init() 通过本机 UDS bootstrap
  -> 远端 sidecar 请求当前 remote daemon 创建 L3 session
  -> 远端 L3 完成 L2 sim child init
  -> HELLO READY 经 MPI 返回 L4
  -> L4 注册 remote callable
  -> L4 向明确 worker_id 提交任务
  -> TASK 经 MPI 到达远端 L3
  -> 远端 L3 调度 L2 sim child
  -> COMPLETION 经 MPI 返回
  -> L4 drain 成功
  -> SHUTDOWN 和所有进程有界退出
```

第一阶段不是 transport echo 测试。只有上述真实 Worker 路径通过，才能进入第二阶段。

### 4.2 Simpler 代码修改范围

核心 runtime 修改进入以下现有模块：

```text
python/simpler/worker.py
  RemoteWorkerSpec 增加 control_transport/sidecar_endpoint/mpi_rank
  新增 sidecar bootstrap/session descriptor 分支
  _activate_remote_sessions() 根据 control_transport 调用对应 attach
  保持 socket 分支原逻辑

src/common/hierarchical/remote_endpoint.h/.cpp
  新增 RemoteL3SidecarTransport
  通过 AF_UNIX command/health lane 连接本机 sidecar proxy
  复用 RemoteL3Endpoint 和完整 SLR3 frame

src/common/hierarchical/worker.h/.cpp
  新增 add_remote_l3_sidecar()
  不修改 add_remote_l3_socket() 的行为

python/bindings/worker_bind.h
  暴露 add_remote_l3_sidecar()

python/simpler/remote_l3_sidecar_proxy.py
  使用 Python 标准库处理本机 UDS、daemon JSON bootstrap 和 session channel
  不执行 MPI，不创建 Worker
```

MPI sidecar 是 L4 remote transport 的组成部分，不是独立 smoke。其代码放在 runtime
所属目录，并独立构建，避免主 wheel 强依赖 MPI：

```text
src/common/hierarchical/mpi_sidecar/
  CMakeLists.txt
  sidecar_main.cc
  sidecar_protocol.h/.cpp
  README.md
```

`tools/mpi_l4_sidecar/` 只允许放启动脚本、hostfile 和本机配置模板，不能放另一套与
Simpler runtime 无关的 TASK/RemoteBuffer 模拟实现：

```text
tools/mpi_l4_sidecar/
  build.sh
  run_1host_sim.sh
  run_2host_sim.sh
  topology.example.json
  verify_no_residue.sh
```

### 4.3 Sidecar 进程模型

每台主机一个 MPI rank，一个 rank 可以承载多个 remote worker session：

```text
rank 0: L4 主机，worker_ids=[]
rank 1: L3 主机 A，worker_ids=[0, 1]
rank 2: L3 主机 B，worker_ids=[2]
```

启动流程：

1. `mpirun` 启动 sidecar executable。
2. launcher 先启动本机 `remote_l3_sidecar_proxy.py` 并等待 internal UDS READY；
   sidecar 在 `MPI_Init` 前连接该 proxy，禁止在 `MPI_Init` 后 fork/exec。
3. sidecar 调用 `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`，只有主线程调用 MPI。
4. `MPI_Allgather` 校验 rank、hostname、role、worker_id 集合和公共配置摘要。
5. 所有 rank 拓扑一致后，rank 0 proxy 才发布 bootstrap UDS READY。
6. L4 `Worker.init()` 连接该 UDS，开始正常 remote session activation。

第一阶段固定采用“launcher 预先启动 proxy”：launcher 必须等待 proxy internal UDS
READY 后再启动 sidecar，并只按本次唯一 JOB_ID 回收它启动的进程和路径，不依赖竞态重试。

### 4.4 Bootstrap 流程

```text
L4 worker.py
  -> 本机 bootstrap UDS
  -> OPEN_SESSION(manifest, target_mpi_rank, remaining_startup_budget)
  -> rank 0 sidecar
  -> MPI P2P OPEN_SESSION
  -> target sidecar
  -> target proxy
  -> 本机 simpler-remote-worker TCP bootstrap
  -> exec simpler-remote-l3-session
  -> inner Worker(level=3).init()
  -> L2 sim child INIT_READY
  -> command/health channel ready
  -> OPEN_SESSION_REPLY 经 MPI 返回
  -> L4 add_remote_l3_sidecar()
  -> C++ transport 连接本机 command/health UDS
  -> HELLO READY
```

现有 daemon manifest 使用 length-prefixed JSON。proxy 必须使用标准 JSON parser 解析和
重建，不允许字符串替换。跨机只传 duration，不传 absolute monotonic timestamp：

```text
remaining = received_startup_remaining - local_elapsed
```

rank 0、target rank 和 target proxy 每一跳都扣减本跳耗时。剩余时间小于等于零时，
必须关闭 channel 并回收已创建 runner。

### 4.5 运行期控制消息

当前 `SLR3` frame 作为 MPI `MPI_BYTE` payload 原样传输：

```text
L4 RemoteL3Endpoint
  -> RemoteL3SidecarTransport
  -> command UDS
  -> rank 0 sidecar
  -> MPI P2P
  -> target sidecar/proxy
  -> runner command socket
  -> remote L3
```

第一阶段至少支持：

```text
HELLO
CONTROL: callable prepare/commit/abort/unregister
TASK
COMPLETION
HEALTH
SHUTDOWN
```

要求：

- 不重新编码 TASK/CONTROL payload。
- 每个 `(session_id, lane)` 保持 FIFO。
- envelope 和 SLR3 header 的 worker_id/session_id/sequence 必须一致。
- 普通消息使用 MPI P2P；禁止在 task 路径使用 collective。
- command/health 使用独立逻辑 lane，health 不能被长 TASK reply 阻塞。
- MPI job/rank 失败时关闭 L4 本机 health UDS，使当前 endpoint failure 路径生效。

### 4.6 第一阶段验证

#### 无 MPI、无设备单元测试

```bash
pytest -q tests/ut/py/test_mpi_sidecar_spec.py
pytest -q tests/ut/py/test_mpi_sidecar_bootstrap.py
ctest --test-dir tests/ut/cpp/build \
  -R '^(test_remote_wire|test_remote_endpoint|test_remote_sidecar_transport)$' \
  --output-on-failure
```

覆盖：

- 新字段默认值保持 socket。
- socket spec 继续执行现有 host:port 校验。
- sidecar spec 的 UDS、rank、worker 映射校验。
- startup deadline 逐跳递减，不重置或倍增。
- UDS frame 截断、超长、sequence 错误和 EOF。
- sidecar attach 仍发生在最后一次本地 fork 后。
- 任一 attach 失败触发当前 Worker.init 原子 rollback。

#### 单机双 rank Simpler 端到端

```bash
bash tools/mpi_l4_sidecar/run_1host_sim.sh topology.local.json
```

这个脚本必须运行两个业务路径：

```text
legacy_socket_sim: L4 -> TCP remote L3 -> L2 sim -> PASS
mpi_sidecar_sim:   L4 -> UDS/MPI remote L3 -> L2 sim -> PASS
```

两条路径使用同一个 callable、输入和 golden，比较：

- 输出结果。
- remote worker_id。
- TASK/COMPLETION sequence。
- 错误类型。
- runner/L2 退出状态。

#### 双机 Simpler 端到端

```bash
bash tools/mpi_l4_sidecar/run_2host_sim.sh topology.2host.json
```

通过条件：

- 日志明确记录 `MPI world READY`、每个 rank/hostname 和 worker 映射。
- L4 只连接本机 UDS，不连接远端 runner command/health TCP 端口。
- TASK 和 COMPLETION 的 payload hash 在发送端、接收端一致。
- 远端 L3 实际创建 L2 sim child 并执行任务。
- 正常 shutdown 后没有 sidecar、proxy、runner、UDS 或 shm 残留。
- 同一构建中的 legacy socket 路径仍通过。

“`mpirun` 已成功运行”的判据不是进程退出码为零，而是同时满足：

```text
MPI world READY
+ L4 HELLO READY
+ L4 -> remote L3 -> L2 TASK/COMPLETION PASS
+ 全部进程和资源正常回收
```

## 5. 第二阶段：完整控制语义和故障边界

第一阶段通过后，补齐当前 socket remote endpoint 已支持的全部控制能力：

```text
ALLOC_REMOTE_BUFFER
FREE_REMOTE_BUFFER
COPY_TO_REMOTE
COPY_FROM_REMOTE
EXPORT_BUFFER（sim）
IMPORT_BUFFER（sim）
RELEASE_IMPORT
动态 callable 注册事务
并发 session/backpressure
```

重点修改和验证：

- sidecar envelope 大小和总在途字节上限。
- 多 worker_id 到同一 rank 的 channel demux。
- 10,000 次 ordered command 不乱序。
- startup deadline、runtime timeout 和 shutdown deadline 分离。
- rank_before_ready、rank_after_hello、drop_completion、stall_command、
  close_health、malformed_envelope 和 budget_exhausted 故障注入。
- 标准 MPI 下任一 rank 失效默认使 MPI 控制面整体失败，不承诺单 endpoint 自动恢复。

验证仍然双跑：

```text
legacy_socket_sim_remote_buffer: PASS
mpi_sidecar_sim_remote_buffer: PASS
```

## 6. 第三阶段：Fabric handle 生命周期闭环

第三阶段增加显式 `transport="a3_fabric"`，`sim` 仍为默认。

### 6.1 描述符

版本化 `A3FabricDescriptor` 至少包含：

```text
owner_worker_id
owner_local_device_id
buffer_id
generation
window_size
access_flags
fabric_handle_type
fabric_handle_bytes
```

描述符不得携带源进程裸 GVA。优先使用当前
`RemoteBufferExport.opaque_transport_descriptor`；如果现有字段不足，再升级 remote
wire，并通过 HELLO capability 协商，保证 v1 socket/sim peer 不受影响。

### 6.2 流程

```text
源 L2 CANN create/map/export
  -> 源 L3 本机 control
  -> 源 sidecar
  -> MPI FABRIC_DESCRIPTOR
  -> 目标 sidecar
  -> 目标 L3 本机 control
  -> 目标 L2 CANN import/reserve/map
  -> importer-local GVA
  -> release/unmap
```

sidecar 只交换 opaque handle 元数据，不调用 CANN。

### 6.3 验证

```bash
# 无设备 codec、capability、假 backend 和错误注入
pytest -q tests/ut/py/test_a3_fabric_descriptor.py

# 双机 A3 真实 handle 生命周期
bash tools/mpi_l4_sidecar/run_2host_fabric_handle.sh topology.2host-a3.json
```

通过条件包括 stale generation、越权 access、重复 import、提前 owner free、rank 失败
回滚，以及循环 100 次无 HBM window/import registry 泄漏。完成后必须再次运行全部
socket/sim RemoteBuffer 测试。

## 7. 第四阶段：Fabric 数据访问闭环

L4 只持有逻辑 `RemoteBufferHandle/RemoteTensorRef`。目标 L3 根据
`worker_id/local_device_id` 路由到 L2，目标 L2 在 task materialization 时完成：

```text
logical remote identity
  -> live import lookup
  -> generation/access/bounds 校验
  -> importer-local GVA
  -> ChipWorker task
```

任务 success、failure、cancel 和 Worker.close 都释放 task 引用；最后一个引用释放后
才允许 unmap/release 和 owner free。

双机最终验证：

```bash
bash tests/st/worker/remote_l3/run_mpi_fabric_tload_e2e.sh \
  topology.2host-a3.json
```

通过条件：

- L4 显式提交到目标 remote worker。
- TASK/COMPLETION 经过 MPI 控制面。
- Fabric handle 通过 MPI 元数据通道交换。
- 设备 payload 不经过 MPI copy。
- 目标 L2 PTO `TLOAD/TSTORE` 结果匹配 peer pattern。
- 所有成功和失败路径无 import/window/task 引用泄漏。
- 同一构建中的 socket/sim L4 -> L3 -> L2 示例继续通过。

## 8. 每阶段统一回归门禁

每个阶段除新增测试外，都运行当前旧路径测试：

```bash
cmake -B /tmp/simpler-regression-ut-cpp -S tests/ut/cpp
cmake --build /tmp/simpler-regression-ut-cpp \
  --target test_remote_wire test_remote_endpoint -j2
ctest --test-dir /tmp/simpler-regression-ut-cpp \
  -R '^(test_remote_wire|test_remote_endpoint)$' --output-on-failure
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q \
  tests/ut/py/test_worker/test_remote_startup_budget.py \
  tests/ut/py/test_remote_l3_lifecycle.py \
  tests/ut/py/test_callable_identity.py
```

合入矩阵：

| 阶段 | 新路径硬结果 | 旧路径硬结果 | 默认行为 |
| --- | --- | --- | --- |
| 一 | L4 -> MPI -> L3 -> L2 sim PASS | socket L4 -> L3 -> L2 PASS | socket/sim |
| 二 | MPI 全控制面和故障注入 PASS | socket RemoteBuffer PASS | socket/sim |
| 三 | A3 handle export/import/release PASS | socket/sim buffer PASS | socket/sim |
| 四 | L4 发起 Fabric TLOAD/TSTORE PASS | socket/sim task PASS | socket/sim |

任一阶段出现以下情况都不能合入：

- 只验证 sidecar echo，没有经过真实 L4/L3/L2。
- 要求现有用户修改配置才能继续使用 socket。
- MPI 不存在时导致默认 Simpler 构建或 import 失败。
- 新路径失败后旧 socket Worker 被关闭或状态污染。
- 为通过测试而绕开当前 root startup deadline、READY barrier 或 rollback。

## 9. 故障和回退

第一、二阶段默认故障域：任一 MPI rank 失败，当前 MPI sidecar world 整体失败。rank 0
关闭所有 L4 本机 command/health UDS，L4 将其转换为现有 endpoint failure，并停止向
这些 remote worker 提交新任务。

回退不需要恢复代码或重新编译：

```python
# 保持或恢复默认值
RemoteWorkerSpec(
    endpoint="10.0.0.8:19073",
    platform="a2a3sim",
    transport="sim",
    control_transport="socket",
)
```

launcher 只清理本次 JOB_ID 的 process group 和临时目录，禁止使用 `killall` 或影响
其他 Simpler job 的全局命令。

## 10. 与现有代码的对应关系

- L4 remote spec、启动 deadline 和 session activation：
  [`worker.py`](simpler/python/simpler/worker.py)
- transport-neutral boundary 和当前 socket transport：
  [`remote_endpoint.h`](simpler/src/common/hierarchical/remote_endpoint.h)
- Worker endpoint attach：
  [`worker.h`](simpler/src/common/hierarchical/worker.h)
- 当前 canonical Remote L3 wire：
  [`remote_wire.h`](simpler/src/common/hierarchical/remote_wire.h)
- 当前 daemon/runner：
  [`remote_l3_worker.py`](simpler/python/simpler/remote_l3_worker.py)、
  [`remote_l3_session.py`](simpler/python/simpler/remote_l3_session.py)
- A3 Fabric handle API 和 TLOAD/TSTORE 参考，仅作为第三、四阶段硬件实现依据：
  [`a3_hccl_smoke`](mpirun-test/tools/a3_hccl_smoke/README.md)
