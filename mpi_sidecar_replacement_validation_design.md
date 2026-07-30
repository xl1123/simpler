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

当前实现基线：`simpler` `d6b73e8c`。第一阶段硬件用例参考
`l4test` `5d8c1dbe` 的 `tools/remote_l4_npu`，但验证入口直接接入 Simpler 的 L4
remote worker 和 MPI sidecar，不复制一套独立通信协议。

## 2. 四项闭环

完整交付需要形成四项闭环：

1. **mpirun 进程闭环**：`mpirun` 在两台机器各启动一个 MPI sidecar rank；机器 A
   另外运行现有 L4 master（不占 MPI rank），共同完成
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
- `transport` 继续表示 remote-buffer 数据/通信 profile，第一阶段真实 NPU 用例仍使用
  `"sim"`，因为输入输出经 `COPY_TO_REMOTE/COPY_FROM_REMOTE` 搬运；它不表示 L2
  simulator。真实 L2 由 `platform="a2a3"` 和 `device_ids=(0, 1)` 选择。不能把
  `control_transport`、buffer transport profile 和 L2 platform 混成同一字段。
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
- 第一阶段保留 L2 sim 用例作为无设备快速回归，但硬门禁必须让两个远端 L3 各自调度
  两个真实 L2 NPU child，不能只在远端 Python dispatcher 返回结果。

## 4. 第一阶段：直接接入 L4 的最小 MPI 控制闭环

### 4.1 阶段目标

第一阶段同时完成最小的 mpirun 进程闭环和 L4 控制闭环：

```text
mpirun 启动至少两个 sidecar rank
  -> rank/host/worker 拓扑 READY
  -> L4 Worker.init() 通过本机 UDS bootstrap
  -> 远端 sidecar 请求当前 remote daemon 创建 L3 session
  -> 两个远端 L3 各自完成两个真实 L2 NPU child init
  -> HELLO READY 经 MPI 返回 L4
  -> L4 注册 remote callable
  -> L4 向明确 worker_id 提交任务
  -> TASK 经 MPI 到达远端 L3
  -> 远端 L3 通过 submit_next_level_group 调度真实 L2 NPU
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

python/simpler/remote_l3_session.py
  真实 platform 的 ALLOC_REMOTE_BUFFER 使用 inner L3 Worker.create_host_buffer()
  使 RemoteTensorRef materialize 为 L2 child 已映射的 host 地址
  a2a3sim 和 childless worker 保留 SharedMemory 回退及 EXPORT_BUFFER(sim) 兼容性
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

`tools/mpi_l4_sidecar/` 只放构建/启动/验证编排和调用生产 Worker API 的端到端入口，
不能放另一套与 Simpler runtime 无关的 TASK/RemoteBuffer 协议模拟实现：

```text
tools/mpi_l4_sidecar/
  build.sh
  launch.py
  npu_e2e_case.py
  run_1host_sim.sh
  run_2host_sim.sh
  run_2host_npu.sh
  run_2host_npu_mpi_only.sh
  topology.example.json
  topology.2host-npu.json
  topology.2host-npu.example.json
  verify_no_residue.sh
```

`npu_e2e_case.py` 是真实 Simpler runtime 的端到端调用入口：它构造生产
`Worker(level=4)`、`RemoteWorkerSpec`、动态 `ChipCallable` 注册、RemoteBuffer 和
`submit_next_level_group()`，不模拟 MPI、SLR3 或 RemoteBuffer 协议。

### 4.3 Sidecar 进程模型

硬件门禁只有两台机器和两个 MPI rank。现有 `npu_e2e_case.py` 是 L4 master；master
运行在机器 A，但不由 `mpirun` 拉起，也不占用第三个 rank：

```text
机器 A / rank 0: L4 master + bootstrap/source + worker_ids=[0] -> NPU 0,1
机器 B / rank 1:                               worker_ids=[1] -> NPU 0,1
worker_map: 0:0;1:1
```

启动流程：

1. launcher 生成 hostfile，并且只调用一次 `mpirun` 启动两个 sidecar rank。
2. 每个 sidecar rank 在 `MPI_Init` 前 fork/exec 本机 `remote_l3_sidecar_proxy.py`，等待
   internal UDS READY 后连接；不由 launcher 通过额外 SSH 启动远端 proxy。
3. sidecar 调用 `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`，只有主线程调用 MPI。
4. `MPI_Allgather` 校验 rank、hostname、role、worker_id 集合和公共配置摘要。
5. 所有 rank 拓扑一致后，rank 0 proxy 才发布 bootstrap UDS READY。
6. L4 `Worker.init()` 连接该 UDS，开始正常 remote session activation。

rank 0 proxy 同时持有 L4 source session 和本机 worker 0 target session。sidecar envelope
v2 使用 `FRAME_L4_TO_L3` / `FRAME_L3_TO_L4` 显式标识方向，不能再用 source rank 推断，
因为本机闭环两端的 rank 都是 0。

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
  -> L2 child INIT_READY（快速门禁为 sim，硬件门禁为两个 a2a3 NPU child）
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
CONTROL: ALLOC/FREE_REMOTE_BUFFER、COPY_TO/FROM_REMOTE
TASK
COMPLETION
HEALTH
SHUTDOWN
```

真实 NPU 用例要求上述 RemoteBuffer control 和动态 CHIP_CALLABLE 注册在第一阶段闭环，
因为 L4 要把输入送到远端 L3 的 L2 可见 host buffer，并在计算完成后取回输出。这些消息
仍作为 canonical SLR3 frame 原样转发，不在 sidecar 中实现业务语义。

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
pytest -q tests/ut/py/test_remote_l3_buffer_allocation.py
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

这条用例是无设备快速门禁，不代替下面的真实 NPU 硬门禁。

#### 双机 sim Simpler 端到端

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

#### 双机、master 位于机器 A 的真实 NPU 硬门禁

拓扑与 `l4test/tools/remote_l4_npu` 的计算任务一致：

```text
机器 A / rank 0: L4 master + remote L3 worker 0 -> NPU 0 + NPU 1
机器 B / rank 1:             remote L3 worker 1 -> NPU 0 + NPU 1
```

操作上是两台机器各开一个 daemon 终端，然后在机器 A 再开一个 master/launcher 终端。
master 逻辑就是 `npu_e2e_case.py` 中的 `Worker(level=4)`。launcher 根据
`mpi.implementation` 和两个 `mpi_host` 生成 hostfile，一次 `mpirun` 拉起两个 sidecar
rank；每个 rank 在本机管理自己的 proxy，再由 launcher 以普通 Python 子进程执行 master。
裸机上的 MPI 实现仍可能在内部使用 SSH，但 Simpler 不再保存 `hosts[].ssh` 或执行 SSH 命令。

两台 NPU 主机先在已加载 CANN 环境的 checkout 中启动现有 daemon：

```bash
source .venv/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
python -m simpler.remote_l3_worker --host 0.0.0.0 --port 19073
```

parent 构建 sidecar、填写拓扑并运行：

```bash
source .venv/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"
bash tools/mpi_l4_sidecar/build.sh
bash tools/mpi_l4_sidecar/run_2host_npu.sh \
  tools/mpi_l4_sidecar/topology.2host-npu.json
```

不执行 socket baseline、让 MPI 成为 daemon 冷启动后的第一个真实 NPU 负载：

```bash
bash tools/mpi_l4_sidecar/run_2host_npu_mpi_only.sh \
  tools/mpi_l4_sidecar/topology.2host-npu.json
```

该入口仍要求两机各有一个预启动的 `remote_l3_worker`，但不会创建默认 L4
socket session，也不会读取 baseline 结果或执行 legacy/MPI 对比。
从两个 daemon 均未启动的环境出发，本阶段还不是单命令流程，因为 launcher 会在
`mpirun` 之前检查两个 daemon 的 TCP 端口。若 daemon 已作为 systemd 服务或后台进程
常驻，则只需在 master 使用一个交互终端运行 MPI-only 脚本。日志中的
`MPI_TARGET_CONNECT_REMOTE_L3` 和 `REMOTE_L3_SESSION_READY` 会显式标记
`daemon_transport=tcp`/`runner_transport=tcp`；这是本阶段仍复用 Remote L3 后端的预期
依赖，不代表默认 L4 socket baseline 被执行。

默认对比入口对同一个 vector group 用例顺序执行：

```text
legacy_socket_npu:
  L4 -> TCP -> remote L3 A/B -> each submit_next_level_group -> 2 x L2 NPU

mpi_sidecar_npu:
  L4 master on A -> rank 0 UDS -> local rank 0 worker / MPI rank 1 worker
     -> each submit_next_level_group -> 2 x L2 NPU
```

MPI-only 入口只执行上面的 `mpi_sidecar_npu`。每条被选中的路径都必须完成动态
`ChipCallable` prepare/commit、每个 worker 六个 RemoteBuffer 的 allocate/copy/free、
两个 NPU group 执行和 golden 校验。默认对比入口还比较两次运行的 worker 映射、
实际输出 SHA-256、expected 值和 max diff；两个入口都会聚合 rank 0/1 日志，校验
sidecar 两侧 TASK/CONTROL/COMPLETION 的 sequence/hash。

这一步证明的是“MPI 控制面驱动了真实跨机 NPU 计算”。输入/输出仍通过 control frame
copy；不包含跨机 NPU buffer import/export，也不等价于第三、四阶段的 Fabric 数据面。

“`mpirun` 已成功运行”的判据不是进程退出码为零，而是同时满足：

```text
MPI world READY
+ L4 HELLO READY
+ legacy socket real-NPU golden PASS
+ MPI sidecar real-NPU golden PASS
+ legacy_vs_mpi output/hash PASS
+ 全部进程和资源正常回收
```

## 5. 第二阶段：完整控制语义和故障边界

第一阶段通过后，补齐当前 socket remote endpoint 的 Fabric-sim 资源语义、压力和故障边界：

```text
EXPORT_BUFFER（sim）
IMPORT_BUFFER（sim）
RELEASE_IMPORT
并发 session/backpressure
大 payload/总在途字节限制
```

基本 ALLOC/FREE/COPY 和动态 callable 注册事务已被真实 NPU 第一阶段用例覆盖；第二阶段
继续做其重复/乱序/超时/回滚故障注入，但不再把“首次可用”推迟到第二阶段。

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
| ---- | ------------ | ------------ | -------- |
| 一 | MPI 控制面驱动两台 remote L3 的真实双 NPU group PASS | 同一真实 NPU 用例 socket PASS | socket/sim |
| 二 | MPI Fabric-sim 资源语义、压力和故障注入 PASS | socket RemoteBuffer PASS | socket/sim |
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
