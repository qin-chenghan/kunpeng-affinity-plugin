# 鲲鹏 GPU 亲和性插件使用指南

## 1. 文档说明

本文档面向使用 vLLM 或 SGLang 部署推理服务的安装、运维和验证人员，说明鲲鹏 GPU 亲和性插件的安装、配置、启用、验证、故障处理和卸载方法。

插件在框架缺少可用 GPU-to-NUMA 结果时，补充逻辑 GPU 到 PCIe BDF、NUMA 节点和 CPU 集合的解析能力，并将经过校验的结果交给框架原有绑定执行器。插件不安装 GPU 驱动或推理框架，不修改操作系统资源限制，也不替代框架执行 CPU 和内存绑定。

## 2. 支持范围

### 2.1 推理框架

| 框架 | 支持的基础版本 | 说明 |
|---|---|---|
| vLLM | `0.23.0`、`0.25.1`、`0.26.0` | 接受以上基础版本的厂商本地版本后缀；其它版本不自动视为兼容。 |
| SGLang | `0.5.18`、`0.5.17.dev386+gc5bd3d7dc` | 普通 Engine 子进程路径；Data Parallel 和 Ray 路径不在当前支持范围内。 |

版本匹配只表示插件允许安装 Hook，不代表任意硬件、Runtime 和启动方式均已验证。部署前仍应完成第 8 章所述的实际服务验证。

### 2.2 设备映射 Provider

| Provider | 适用条件 |
|---|---|
| `auto` | 自动选择唯一满足当前完整设备批次的 Provider，推荐用于正常部署。 |
| `vllm-platform-pci` | vLLM 平台接口能够完整返回所有可见逻辑设备的 BDF。 |
| `sglang-runtime-pci` | SGLang 所在 Runtime 能够返回直接 BDF 或可关联的设备身份。 |
| `iluvatar-runtime-pci` | Runtime 能取得逻辑设备 UUID，且 `ixsmi` 能返回 UUID 与 BDF 清单。 |
| `ascend-sysfs-pci` | 已设置有序的 `ASCEND_RT_VISIBLE_DEVICES`，且 Ascend 驱动 sysfs 映射表可读。 |

## 3. 运行前提

所有检查和安装操作都应在实际启动推理服务的容器或 Python 环境中执行，并使用与服务相同的用户、解释器、设备可见性和资源限制。

### 3.1 通用依赖

- Linux 多 NUMA 节点系统；单 NUMA 节点系统没有自动 NUMA 优化目标。
- Python 3.10 或更高版本。
- Python 包 `packaging>=21`。
- 受支持的 vLLM 或 SGLang，以及与设备匹配的驱动、Runtime 和框架扩展。
- 可读的 PCI、NUMA 和 CPU sysfs。
- 可加载且可用的 `libnuma.so` 或 `libnuma.so.1`。
- vLLM 和 SGLang V2 路径需要 `numactl` 位于 `PATH`；SGLang V1 不需要该命令，但仍需要 libnuma 和内存策略权限。
- 容器或服务进程必须允许目标 CPU、内存节点和相关系统调用，并允许创建框架子进程。
- vLLM 多进程路径使用 `spawn`；SGLang V2 还要求可写且可执行的临时目录。

插件不会自动安装上述依赖。

### 3.2 Provider 依赖

**Iluvatar**

- 框架或 Torch Runtime 能取得当前逻辑设备的 UUID。
- `ixsmi` 可执行，其依赖的驱动和 Runtime 库在服务环境中可用。
- `ixsmi` 查询结果包含设备索引、UUID 和 PCI BDF；插件按 UUID 关联设备，不使用物理枚举顺序推断逻辑设备。

**Ascend**

- `ASCEND_RT_VISIBLE_DEVICES` 明确给出有序的可见物理设备 ID。
- 驱动导出的 `devdrv_sysfs_bdf_to_devid` sysfs 属性可读，并完整覆盖可见设备。
- Provider 不依赖 `npu-smi`、DCMI 或 `/dev/davinciN` 后缀推断身份。

### 3.3 前置检查

```bash
<python> --version
<python> -m pip check
command -v numactl
numactl --hardware
numactl --show
```

SGLang V1 可省略 `numactl` 命令检查。检查结果只证明当前进程具备基础条件，不能代替对真实 EngineCore 或 Worker 的绑定验证。

## 4. 安装与卸载

### 4.1 安装

将插件安装到启动推理框架的同一个 Python 环境。使用正式 wheel 制品时执行：

```bash
<python> -m pip install <wheel-file>
```

从交付源码安装时，在源码根目录执行：

```bash
<python> -m pip install --no-deps --no-build-isolation .
```

源码开发或现场调试需要 editable 安装时执行：

```bash
<python> -m pip install --no-deps --no-build-isolation -e .
```

`--no-deps` 不会补装缺失依赖。安装前应确认 Python、`packaging`、框架和系统 NUMA 组件已经就绪。

### 4.2 检查安装结果

```bash
<python> - <<'PY'
from importlib.metadata import entry_points, version

print("plugin_version:", version("kunpeng-affinity-plugin"))
for group in ("vllm.general_plugins", "sglang.srt.plugins"):
    matches = [ep.value for ep in entry_points(group=group) if ep.name == "kunpeng_affinity"]
    print(group, matches)
PY
```

两个 entry-point 组都应显示插件注册函数。该检查只能证明包元数据可见，不能证明框架已执行插件入口或完成绑定。

### 4.3 卸载

停止使用该插件的新服务进程后执行：

```bash
<python> -m pip uninstall kunpeng-affinity-plugin
```

插件不修改框架源码。卸载后新启动的框架进程恢复为未安装插件时的行为。

## 5. 配置

### 5.1 通用配置

| 环境变量 | 可选值 | 默认值 | 说明 |
|---|---|---|---|
| `KUNPENG_AFFINITY_MODE` | `off`、`auto`、`strict` | `auto` | 控制插件关闭、自动回退或严格失败。 |
| `KUNPENG_AFFINITY_PROVIDER` | `auto` 或已注册 Provider 名称 | `auto` | 选择设备映射来源。 |
| `KUNPENG_AFFINITY_CPU_POLICY` | `node`、`exact` | `node` | 当前框架适配器仅支持 `node`；不要在生产配置中使用 `exact`。 |
| `KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL` | `error`、`summary`、`detail` | `summary` | 控制插件诊断级别。 |
| `KUNPENG_AFFINITY_CONFIG` | UTF-8 JSON 文件 | 未设置 | 指定可选配置文件。环境变量覆盖文件中的同名字段。 |
| `KUNPENG_AFFINITY_IXSMI` | 可执行文件路径 | `PATH` 中的 `ixsmi` | 指定 Iluvatar 查询工具。 |

推荐的常规生产配置为：

```bash
export KUNPENG_AFFINITY_MODE=auto
export KUNPENG_AFFINITY_PROVIDER=auto
export KUNPENG_AFFINITY_CPU_POLICY=node
export KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL=summary
```

### 5.2 JSON 配置文件

配置文件根节点必须是 JSON 对象，只允许以下字段：

```json
{
  "mode": "auto",
  "provider": "auto",
  "cpu_policy": "node",
  "diagnostic_level": "summary"
}
```

未知字段、重复字段、非法值、非 UTF-8 内容或非对象根节点都会导致配置失败。若设置 `KUNPENG_AFFINITY_MODE=off`，插件会在读取可选配置文件前退出。

### 5.3 运行模式

| 模式 | 行为 |
|---|---|
| `off` | 不安装或不执行插件 Hook。该模式不负责关闭框架自身的绑核功能。 |
| `auto` | 优先保留显式配置和有效原生结果；允许通用发现。插件发现失败时记录原因并尽量保持框架原有行为。 |
| `strict` | 插件发现或校验失败时报告错误；框架自身配置和绑定执行错误仍由框架处理。 |

## 6. vLLM 使用方法

### 6.1 启动服务

vLLM 必须显式启用自身的 NUMA 绑定开关。不要同时提供显式 `numa_bind_nodes`，否则插件会保留该显式配置而不进行自动发现。

```bash
export KUNPENG_AFFINITY_MODE=auto
export KUNPENG_AFFINITY_PROVIDER=auto
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve <model> --numa-bind <other-vllm-options>
```

如果部署已经设置 vLLM 插件过滤变量，必须允许 `kunpeng_affinity` 被加载。安装插件本身不会自动添加 `--numa-bind`。

### 6.2 vLLM 决策顺序

```text
用户显式 NUMA 节点
  -> 有效的 vLLM 原生结果
  -> Provider BDF 映射与 Linux sysfs 通用分析
  -> vLLM 原有 numactl 执行路径
```

插件只自动补充缺失的 NUMA 节点列表，并保留用户显式提供的 CPU 列表。

### 6.3 强制通用路径诊断

`KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1` 只用于隔离验证：它跳过 vLLM 原生 NUMA 查询，强制验证 Provider 和 Linux 拓扑路径。该开关不会启用 `--numa-bind`，不会选择 Provider，并且通用发现失败时会报错，即使模式为 `auto`。

正常部署应保持该变量未设置：

```bash
unset KUNPENG_AFFINITY_VLLM_FORCE_GENERIC
```

## 7. SGLang 使用方法

### 7.1 启动服务

```bash
export KUNPENG_AFFINITY_MODE=auto
export KUNPENG_AFFINITY_PROVIDER=auto
export SGLANG_PLUGINS=kunpeng_affinity
export SGLANG_AUTO_NUMA_BIND=1
export SGLANG_NUMA_BIND_V2=1

sglang serve <model> <other-sglang-options>
```

`SGLANG_NUMA_BIND_V2=1` 使用 SGLang 的外部 `numactl` 路径。若明确选择 SGLang V1，可设置为 `0`；V1 使用进程内 libnuma，不需要 `numactl` 命令，但仍需要 libnuma 和内存策略权限。V1 与 V2 是显式执行模式选择，插件不会在二者之间自动切换。

以下情况不会使用通用回退：

- `SGLANG_AUTO_NUMA_BIND=0`；
- 已显式配置 NUMA 节点；
- 启用了 `SGLANG_SET_CPU_AFFINITY=1`，因为该功能会覆盖 NUMA 产生的 CPU affinity；
- 使用尚未支持的 Data Parallel 或 Ray 启动路径。

### 7.2 SGLang 决策顺序

```text
用户显式 NUMA 节点
  -> SGLang 原生查询结果
  -> Provider BDF 映射与 Linux sysfs 通用分析
  -> SGLang 原有绑定执行路径
```

## 8. 生效验证

生产部署前应按以下顺序验证，不能仅凭安装成功判断插件已经生效。

1. **包可见**：服务解释器可以导入插件，并能看到对应 entry point。
2. **插件已加载**：服务日志出现插件 Hook 安装或注册信息。
3. **通用路径已选择**：需要通用回退时，日志出现 Provider、逻辑设备、BDF 和 NUMA 节点选择信息。
4. **框架已执行绑定**：检查实际 EngineCore、Worker 或 Engine 子进程的 `Cpus_allowed_list`。
5. **内存策略已执行**：使用目标环境认可的工具或进程接口确认实际 memory policy；仅检查允许的内存节点集合不足以证明已应用策略。
6. **服务可用**：执行健康检查和固定推理请求，确认服务启动和请求结果正常。

自动模式下，插件可能因条件不足而安全跳过，因此“服务成功启动”不等于“插件完成绑定”。

## 9. 常见问题

### 9.1 日志中没有插件加载信息

- 确认插件安装在服务实际使用的 Python 环境中。
- 确认框架版本在支持范围内。
- 检查框架插件过滤设置是否允许 `kunpeng_affinity`。
- 确认 `KUNPENG_AFFINITY_MODE` 不是 `off`。

### 9.2 插件已加载但没有通用 NUMA 选择日志

- vLLM 必须启用 `--numa-bind`，且不能已经显式提供 NUMA 节点。
- SGLang 必须启用 `SGLANG_AUTO_NUMA_BIND=1`。
- 框架原生结果有效时，插件会保留原生路径，这是正常行为。
- 检查当前启动方式是否属于尚未支持的 DP、Ray 或其它独立路径。

### 9.3 提示找不到绑定执行器

- vLLM 和 SGLang V2 需要 `numactl` 位于服务进程的 `PATH`。
- SGLang V1 不需要命令，但仍要求 libnuma 可加载且系统允许内存策略操作。
- 安装在宿主机上的工具不能补偿容器内缺失的工具或库。

### 9.4 Provider 无法映射设备

- Iluvatar：确认逻辑设备 UUID 可查询，`ixsmi` 可执行并返回 UUID/BDF；必要时设置 `KUNPENG_AFFINITY_IXSMI`。
- Ascend：确认 `ASCEND_RT_VISIBLE_DEVICES` 顺序正确，驱动 sysfs 映射表可读且覆盖全部可见设备。
- 不要使用物理设备枚举顺序替代逻辑设备映射。

### 9.5 内存策略或绑核被拒绝

- 检查容器 cpuset 和允许的内存节点。
- 检查 seccomp、系统调用权限和临时目录执行权限。
- 不要通过扩大容器权限或资源范围来掩盖错误映射；应按部署安全策略提供最小必要权限。

## 10. 使用边界

- 插件只在框架启动或子进程配置阶段运行拓扑发现，不进入逐请求推理路径。
- 插件 `off` 与框架“完全不绑核”不是同一概念；性能基线必须同时显式关闭框架自身的绑定。
- 插件日志证明的是加载、发现和决策过程；实际绑定必须检查目标子进程。
- 插件不承诺固定性能收益。性能结论必须来自相同模型、并行度、请求集、框架参数和系统负载下的重复 A/B 测试。
- 对未列入支持范围的框架版本、设备 Runtime 和启动路径，应先完成独立兼容性验证再投入使用。
