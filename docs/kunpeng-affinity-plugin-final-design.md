# 鲲鹏 GPU 亲和性插件系统设计说明书

## 文档状态

本文档为鲲鹏 GPU 亲和性插件最终版设计文档的增量稿。本文描述产品目标、应用场景和产品规格；未在目录中列出的章节暂不在本文档中占位。具体实现细节、测试记录和版本变更将在后续按照既定目录继续补充。

# 1 系统设计说明书

## 1.4 Keywords 关键词

| English | 中文 |
|---|---|
| GPU CPU affinity | GPU CPU 亲和性 |
| NUMA-aware binding | NUMA 感知绑核 |
| PCIe topology | PCIe 拓扑 |
| PCI Bus-Device-Function | PCI 总线-设备-功能地址 |
| Linux sysfs | Linux sysfs 文件系统 |
| Runtime Provider | 运行时设备映射 Provider |
| vLLM and SGLang integration | vLLM 与 SGLang 集成 |
| CPU and memory affinity | CPU 与内存亲和性 |

## 1.5 Abstract 摘要

### 中文摘要

鲲鹏 GPU 亲和性插件是一款面向鲲鹏 CPU 平台推理服务器的独立 Python 插件。插件为 vLLM、SGLang 等推理框架补充 GPU 到 CPU/NUMA 亲和性信息的自动发现能力，解决部分 GPU 设备运行时或推理框架无法直接完成 GPU 到 NUMA 节点映射的问题。

插件以框架当前可见的逻辑 GPU 为输入，首先通过框架原生接口或目标设备 Runtime Provider 获取可信的 PCIe Endpoint BDF，再基于 Linux sysfs 遍历 PCIe 父设备链，解析 NUMA 归属和可用 CPU 集合。插件只负责设备身份映射、拓扑分析、结果校验和缺失配置的决策提交，不替代 GPU 驱动、设备 Runtime 或框架已有的绑核执行器。完成分析后，插件将结果交给框架原有的进程启动和 CPU/内存绑定链路执行。

插件遵循“用户显式配置优先、框架原生查询次之、通用 Linux 拓扑分析回退”的决策顺序。当设备身份、PCIe 路径、NUMA 证据或资源权限无法被可靠证明时，插件不猜测绑定目标；在自动模式下允许框架继续执行，在严格模式下报告错误。多设备场景采用整批校验和原子提交，避免提交不完整或相互矛盾的绑定结果。

### English Abstract

Kunpeng GPU Affinity Plugin is an independent Python plugin for inference servers based on Kunpeng CPUs. It extends inference frameworks such as vLLM and SGLang with automatic discovery of GPU-to-CPU and GPU-to-NUMA affinity information. The plugin addresses environments where the GPU runtime or the inference framework cannot directly resolve the NUMA node associated with a visible logical GPU.

The plugin takes the framework-visible logical GPU context as input. It first obtains a trusted PCIe endpoint BDF from a framework-native interface or a device-specific Runtime Provider. It then traverses the PCIe parent path exposed by Linux sysfs to resolve NUMA evidence and the usable CPU set. The plugin is responsible for device identity mapping, topology analysis, result validation, and completion of missing affinity configuration. It does not replace the GPU driver, the device Runtime, or the framework's existing binding executor. The validated result is handed back to the framework's original process-launch and CPU/memory binding path.

The decision order is explicit user configuration, framework-native discovery, and generic Linux topology fallback. When device identity, PCIe path, NUMA evidence, or resource permissions cannot be proven reliably, the plugin never guesses a binding target. Automatic mode allows the framework to continue when a plugin discovery failure is recoverable; strict mode reports the failure. Multi-device resolution uses all-or-nothing validation and atomic submission so that incomplete or inconsistent binding results are never committed.

## 1.6 List of Abbreviations 缩略语清单

| 缩略语 | English full name | 中文解释 |
|---|---|---|
| BDF | Bus-Device-Function | PCI 设备的总线、设备和功能地址，用于标识 PCIe Endpoint。 |
| CPU | Central Processing Unit | 中央处理器。 |
| GPU | Graphics Processing Unit | 图形处理器；本文泛指用于推理计算的 GPU 或兼容加速设备。 |
| NUMA | Non-Uniform Memory Access | 非一致性内存访问架构。 |
| PCI | Peripheral Component Interconnect | 外设部件互连总线标准。 |
| PCIe | Peripheral Component Interconnect Express | PCI 高速串行总线及其设备互连结构。 |
| PID | Process Identifier | 进程标识符。 |
| SPI | Service Provider Interface | 服务提供者接口；本文指设备映射 Provider 遵循的接口边界。 |
| TP | Tensor Parallelism | 张量并行。 |
| DP | Data Parallelism | 数据并行。 |
| API | Application Programming Interface | 应用程序编程接口。 |
| CLI | Command-Line Interface | 命令行接口。 |
| CANN | Compute Architecture for Neural Networks | 昇腾神经网络计算架构软件栈。 |
| NVML | NVIDIA Management Library | NVIDIA 管理库；本文不将其作为插件的通用依赖。 |
| sysfs | Linux system filesystem | Linux 内核向用户空间暴露设备、内核和系统信息的虚拟文件系统。 |
| UUID | Universally Unique Identifier | 通用唯一标识符；在支持的设备 Runtime 中用于稳定识别设备。 |
| V1/V2 | Version 1 / Version 2 | 框架或其 NUMA 执行路径的版本化实现，具体含义以对应框架版本为准。 |

# 2 简介

随着 GPU 推理服务向多卡、多 NUMA 节点和复杂 PCIe 拓扑发展，GPU 与 CPU 之间的物理连接关系会直接影响 CPU 侧调度、主机到设备的数据传输以及内存访问局部性。推理框架通常已经具备 CPU affinity、NUMA memory policy 和 GPU 设备管理能力，但其自动 GPU-to-NUMA 查询往往依赖特定厂商 Runtime 或框架已经适配的后端。当目标设备能够运行推理任务、但缺少对应的原生 NUMA 查询实现时，框架可能无法自动选择与 GPU 拓扑匹配的 CPU 和内存节点。

本插件在框架的 NUMA 查询与原生绑定执行之间提供补充能力。它将逻辑 GPU 身份映射与 Linux 拓扑分析分为两个边界：

1. **BDF 之前**：由框架适配器和设备 Runtime Provider 根据当前进程的可见性规则建立逻辑 GPU 到 PCIe Endpoint BDF 的可信映射。
2. **BDF 之后**：由框架无关的 Linux 拓扑分析器读取 PCI sysfs，遍历 Endpoint 到 Root Port 或 Host Bridge 的父设备链，解析 NUMA 证据并计算受资源约束的 CPU 集合。

插件不通过物理设备枚举顺序猜测逻辑设备，不把 GPU 名称、设备节点后缀或非唯一序列号作为跨层身份锚点。设备映射必须覆盖当前可见设备，并满足唯一性、可重复性和可验证性要求。

插件最终提交的是框架能够消费的 NUMA 或 CPU 亲和性配置。真正的 `numactl`、`libnuma`、`sched_setaffinity` 或框架内部绑定操作仍由目标框架和其执行器完成。因此，插件安装成功、Provider 返回成功或拓扑分析成功，都不能单独证明推理子进程已经完成绑定；完整验证还需要检查框架 Hook 调用、子进程 CPU affinity 和内存策略。

## 2.1 设计目标

本产品的设计目标如下：

- 以独立 Python 包接入目标推理框架，不修改框架源码或重新发布框架二进制。
- 为框架可见的逻辑 GPU 提供可验证的 BDF 映射能力。
- 统一支持 GPU 直连 CPU、单级 PCIe Switch 和多级 PCIe Switch 的父链分析。
- 使用 Linux sysfs 提供的设备、NUMA 和 CPU 信息计算可用亲和性结果。
- 保持用户显式配置和框架原生查询的优先级，不覆盖已有有效配置。
- 在当前进程允许的 CPU 集合和内存节点范围内生成结果，不扩大容器或进程资源边界。
- 对多设备结果执行全量校验，禁止提交部分成功的设备列表。
- 在身份、拓扑或执行前置条件不足时失败关闭，不生成猜测性绑定。

## 2.2 产品边界

插件负责：

- 框架插件入口和版本/接口兼容性检查；
- 逻辑设备上下文构造和 visibility 语义保持；
- Runtime Provider 注册、设备身份查询和 BDF 映射；
- Linux PCIe/NUMA/CPU 拓扑读取与证据一致性校验；
- 原生结果分类、通用回退决策和批量结果提交；
- 结构化日志、错误码和只读诊断工具。

插件不负责：

- GPU 驱动、GPU Runtime、CANN/CoreX 等厂商软件栈的安装和初始化；
- 修改 Linux 内核、调度器、cgroup、容器安全策略或主机 NUMA 配置；
- 实现或替代框架已有的进程启动、`numactl`、`libnuma` 或 CPU affinity 执行逻辑；
- 保证 CPU 核独占、消除所有跨 NUMA 访问或对任意框架版本提供兼容；
- 在框架绑核开关关闭时擅自打开绑核；
- 对 Data Parallel、Ray 或其他独立框架启动路径作未经验证的能力承诺。

# 3 应用场景分析

## 3.1 适用场景

### 3.1.1 多 NUMA 节点 GPU 推理服务器

服务器包含多个 CPU NUMA 节点，GPU 通过 PCIe 直连或 PCIe Switch 连接到不同的 CPU 根端口。推理框架启动多卡 Worker 或 EngineCore 子进程时，需要为每个可见 GPU 选择其拓扑邻近的 NUMA 节点和 CPU 集合。插件从 BDF 出发读取真实 PCIe 父链并计算结果，适合该类拓扑明确且 Linux sysfs 可读的服务器。

### 3.1.2 框架原生 GPU-to-NUMA 查询缺失或返回空结果

目标 GPU 可以通过框架完成推理，但框架原生自动查询不支持该设备、返回空结果或返回结果不完整。只要 Provider 能建立完整的逻辑 GPU 到 BDF 映射，插件可以将原生查询结果分类为允许回退的情况，并使用通用 Linux 拓扑分析补齐 NUMA 信息。

### 3.1.3 Iluvatar 等需要 Runtime 身份桥接的设备

当框架能够取得逻辑 GPU UUID，但不能直接取得 BDF 时，可由 Iluvatar Runtime Provider 使用只读设备工具建立 UUID 到 PCI BDF 的关联。Provider 不使用工具的物理枚举顺序作为逻辑映射，适合存在可稳定查询 UUID/BDF 关系的环境。

### 3.1.4 Ascend 等可从驱动 sysfs 读取设备映射的设备

当设备 Runtime 通过可见设备列表和驱动导出的 sysfs 属性提供物理设备到 BDF 的关系时，可由 Ascend sysfs Provider 解析该关系，再复用通用 PCIe/NUMA 分析。该场景要求可见设备顺序和驱动映射表均可读取，不能用设备枚举顺序替代显式 visibility 语义。

### 3.1.5 框架原生结果与通用结果的可控回退

在生产或验证环境中，用户可能已经显式设置 NUMA 节点或 CPU 集合，也可能希望优先使用框架原生结果。插件支持保留显式配置、校验有效原生结果，并只在满足回退条件时运行通用分析。自动模式用于尽量保持服务可启动，严格模式用于验证或对绑定正确性有强约束的环境。

## 3.2 不适用或暂不承诺的场景

以下场景不属于当前产品的无条件支持范围：

- 单 NUMA 节点服务器，因不存在可利用的 NUMA 局部性选择；
- 无法建立逻辑 GPU 到唯一 BDF 映射的设备或 Runtime；
- PCI sysfs 不可读、PCIe 父链不完整或 NUMA 证据互相矛盾的环境；
- 容器 cpuset、内存节点限制、seccomp 或权限策略阻止框架执行绑定的环境；
- 框架未加载插件、版本/API 不兼容或目标启动路径没有调用插件加载器；
- SGLang Data Parallel、Ray 等尚未完成独立生命周期验证的启动路径；
- 需要插件自行替代驱动、Runtime 或框架绑定执行器的场景；
- 以插件自动绑定取代 CPU 核独占、实时调度或完整资源隔离的场景。

## 3.3 典型调用链

```text
框架启动
  -> 插件入口加载
  -> 读取框架配置与进程上下文
  -> 判断是否允许自动 NUMA 查询
  -> 保留显式配置或校验框架原生结果
  -> 构造当前可见 GPU 的有序设备批次
  -> Runtime Provider 获取逻辑 GPU 对应 BDF
  -> Linux sysfs 遍历 PCIe 父链
  -> 解析 NUMA 节点并计算受限 CPU 集合
  -> 全量校验并原子提交缺失配置
  -> 框架原有执行器启动子进程并完成绑定
```

该调用链中的最后一步不属于插件实现。插件的成功日志只能证明插件完成了相应决策，必须结合框架执行日志和子进程状态才能判断实际绑定是否生效。

# 4 产品规格定义

## 4.1 功能规格

| 编号 | 功能 | 规格要求 |
|---|---|---|
| F-01 | 框架接入 | 通过目标框架提供的插件入口加载，不修改框架源码。 |
| F-02 | 可见设备语义 | 按框架当前进程的逻辑设备可见性构造有序设备批次，不使用整机物理枚举顺序替代逻辑顺序。 |
| F-03 | 设备映射 | 将每个可见逻辑 GPU 映射为唯一、规范化且可验证的 PCIe Endpoint BDF。 |
| F-04 | Provider 扩展 | 通过独立 Provider 边界接入不同 GPU Runtime，新增 Provider 不修改通用拓扑分析器。 |
| F-05 | PCIe 分析 | 从 Endpoint BDF 开始读取并遍历 Linux sysfs 中的真实 PCIe 父设备链。 |
| F-06 | NUMA 解析 | 优先使用 Endpoint NUMA 证据，并结合祖先设备属性进行一致性校验；无法得到唯一可靠结果时失败关闭。 |
| F-07 | CPU 集合 | 计算 `NUMA node CPUs ∩ online CPUs ∩ current process allowed CPUs`，不得扩大进程允许范围。 |
| F-08 | 决策优先级 | 遵循显式配置、框架原生结果、插件通用结果的优先级。 |
| F-09 | 批量提交 | 多卡结果必须全部可提交后一次性写入；任何设备失败都不得提交部分列表。 |
| F-10 | 框架执行复用 | 将结果交给框架现有的 NUMA/CPU 绑定执行路径，不在插件中重复实现绑定器。 |
| F-11 | 模式控制 | `off` 不安装或执行插件 Hook；`auto` 对可恢复的发现失败保持框架启动；`strict` 将发现失败报告为错误。 |
| F-12 | 诊断能力 | 提供只读拓扑和 Provider 诊断信息，包括设备身份、BDF、NUMA 证据、CPU 交集和失败原因。 |

## 4.2 配置规格

插件配置至少包括以下逻辑项：

| 配置项 | 作用 | 默认行为 |
|---|---|---|
| `KUNPENG_AFFINITY_MODE` | 控制插件关闭、自动和严格模式。 | `auto`。 |
| `KUNPENG_AFFINITY_PROVIDER` | 选择 `auto` 或指定 Runtime Provider。 | `auto`。 |
| `KUNPENG_AFFINITY_CPU_POLICY` | 选择 NUMA 节点 CPU 集合策略。 | 使用框架和插件约定的节点策略。 |
| `KUNPENG_AFFINITY_IXSMI` | 指定 Iluvatar 只读查询工具路径。 | 使用 `PATH` 中的 `ixsmi`。 |
| `KUNPENG_AFFINITY_VLLM_FORCE_GENERIC` | 在隔离验证中跳过 vLLM 原生 NUMA 查询，强制走插件通用发现。 | 未设置。 |
| `SGLANG_AUTO_NUMA_BIND` | 控制 SGLang 自动 NUMA 查询是否参与。 | 遵循目标 SGLang 版本默认值。 |
| `SGLANG_NUMA_BIND_V2` | 选择 SGLang 的外部 `numactl` 或进程内 NUMA 执行路径。 | 遵循目标 SGLang 版本默认值。 |

`FORCE_GENERIC` 只控制 vLLM 的发现路径，不负责打开框架绑核开关，也不选择 Provider。显式 `numa_bind_nodes`、关闭框架 `numa_bind` 或插件 `off` 均优先于该诊断开关。该开关适合隔离验证 generic 路径，不应被理解为通用生产策略开关。

## 4.3 兼容性规格

兼容性必须同时满足版本、模块、函数签名、加载时机和真实调用路径要求。仅版本号匹配或 entry point 元数据存在，不构成完整兼容证明。

当前设计基线如下：

| 框架 | 版本基线 | 适配范围和状态 |
|---|---|---|
| vLLM | `0.23.0` | 第一阶段完整开发和验收基线。 |
| vLLM | `0.25.1` | 为 Ascend 环境纳入的兼容基线，需按目标 Runtime 复验完整服务链路。 |
| vLLM | `0.26.0` | 已用于通用 spawn/fallback 诊断的兼容基线。 |
| SGLang | `0.5.18` | 当前 SGLang 适配基线。 |
| SGLang | `0.5.17.dev386+gc5bd3d7dc` | 已观察到的兼容构建，需按部署镜像独立确认。 |

厂商本地版本后缀只有在其上游基础版本被纳入支持矩阵时才可进入验证；不应根据版本字符串相近就推断兼容。每个目标硬件、Runtime、框架和启动路径仍需完成对应的功能与执行链验证。

## 4.4 部署规格

部署环境至少应具备：

- 与服务相同的 Python 解释器、用户和包环境；
- 目标 vLLM 或 SGLang 及其匹配的 GPU 驱动、Runtime 和框架扩展；
- 可加载的 `libnuma`，以及目标路径所需的 `numactl`；
- 可读的 PCI、NUMA 和 CPU sysfs；
- Provider 所需的设备身份接口，例如 Iluvatar 的只读 `ixsmi` 或 Ascend 的 sysfs 映射表；
- 目标 CPU、内存节点和系统调用未被容器资源或安全策略阻止；
- 框架子进程启动路径可用，vLLM NUMA wrapper 所需的 spawn、解释器和 shell 条件满足。

插件安装器不自动安装 GPU 驱动、厂商 Runtime、推理框架、`libnuma` 或 `numactl`。这些依赖必须由部署镜像或主机环境按实际发行版和框架要求提供。

## 4.5 非功能规格

| 类别 | 规格要求 |
|---|---|
| 正确性 | 不能在设备身份或 NUMA 证据不足时生成猜测性结果。 |
| 安全性 | 默认只读取设备、sysfs 和框架状态；不修改驱动、主机拓扑或容器资源。 |
| 可回退性 | 自动模式下可恢复的插件发现失败不应阻断原框架启动；原框架执行错误不得被插件吞掉。 |
| 可观测性 | 日志应区分插件加载、Provider 映射、拓扑分析、决策来源和框架执行结果。 |
| 可维护性 | 框架类型隔离在 Adapter，设备 Runtime 隔离在 Provider，通用拓扑逻辑不依赖厂商库。 |
| 资源开销 | 拓扑发现发生在框架子进程启动配置阶段，不应对每个推理请求重复执行。 |
| 一致性 | 多设备结果采用同一 visibility 和拓扑快照验证，发现期间发生变化时不得提交。 |
| 兼容性 | 对未验证的框架版本、启动路径和并行模式必须显式拒绝、跳过或标记待验证。 |

# 5 系统设计方案概述

本系统由框架适配层、设备身份 Provider、通用 Linux 拓扑分析层和框架执行衔接层组成。各层沿一条有序流程协作：框架适配层读取当前启动上下文和可见设备，Provider 将逻辑设备映射为可信的 PCIe Endpoint BDF，拓扑分析层从 BDF 读取 Linux sysfs 中的 PCIe 父链和 NUMA 信息，决策层校验并生成框架需要的 NUMA 配置，最后复用框架已有的进程启动与绑定执行器。

```text
框架插件入口
  -> 可见逻辑设备上下文
  -> Runtime Provider 设备身份映射
  -> BDF 规范化与 PCIe/NUMA 拓扑分析
  -> 结果校验与模式决策
  -> 框架原有的 CPU/内存绑定执行
```

系统的关键解耦边界如下：

- **框架适配层**只处理 vLLM、SGLang 的入口、版本、调用签名和配置语义。
- **Runtime Provider**只处理特定设备 Runtime 提供的逻辑设备到 BDF 映射，不负责 PCIe 父链分析和实际绑核。
- **通用拓扑分析层**只依赖 Linux sysfs、CPU 和 NUMA 信息，不依赖具体 GPU 厂商库。
- **执行衔接层**只把已经校验的结果交给框架原有执行器，不在插件内重复实现 `numactl`、`libnuma` 或 `sched_setaffinity`。

系统采用显式配置优先、框架原生结果次之、插件通用分析回退的策略。所有可见设备必须作为一个完整批次校验；批次中任一设备无法获得可靠结果时，不提交部分结果。插件只在框架自身已经启用相应绑核路径时提供补充信息，不自行打开框架绑核开关。

# 6 系统级设计约束

## 6.1 license 约束

- 插件及其正式交付文档必须随正式仓库明确许可证和版权归属。
- 当前开发仓未包含 `LICENSE`、`NOTICE` 或 `COPYING` 文件，因此本设计不虚构具体许可证；正式发布前必须补齐并核对许可证文件。
- 插件只调用目标框架、GPU Runtime、Linux 系统接口和只读设备查询工具，不复制或修改这些外部软件的实现。外部组件仍受其各自许可证和交付条款约束。
- 依赖清单、许可证声明和第三方版权信息应与正式交付版本保持一致。插件安装过程不应绕过或改变外部组件的许可校验。

## 6.2 性能约束

- 设备身份和 PCIe/NUMA 拓扑发现应发生在框架启动或子进程配置阶段，不得在每个推理请求路径上重复执行。
- 插件不得引入常驻线程、请求级轮询或与模型输入输出相关的额外处理。
- 拓扑分析的规模主要随当前可见设备数和 PCIe 父链深度增长；实现应复用单次读取结果，避免无意义的重复 sysfs 访问。
- 插件不承诺固定的 TTFT、TPOT、吞吐或端到端时延收益。性能结果必须在模型、并行度、请求集、框架配置和绑核状态一致的条件下进行对照，并区分预热、缓存和系统负载影响。
- 插件自身的启动开销和服务性能影响属于验收项目；功能通过不等于性能达标。

## 6.3 可靠性/可用性/Safety

- 设备身份、BDF、PCIe 父链或 NUMA 证据不完整、冲突或无法读取时，不生成猜测性绑定结果。
- 多卡结果采用全量校验和原子提交，禁止因为部分设备成功而提交部分绑定列表。
- `auto` 模式下，插件可恢复的发现失败应记录原因并保持框架原有启动行为；`strict` 模式下应将该失败明确报告给调用方。
- 插件不得吞掉框架配置错误、设备索引错误、绑定执行器错误或子进程启动错误。这些错误仍由框架原有路径处理。
- 插件加载、拓扑分析和配置提交失败不得修改 GPU 驱动状态、重置设备或中断已有推理进程。
- 拓扑信息只在一次决策中作为一致快照使用；发现过程中发生设备可见性或拓扑变化时，应放弃本次提交并报告失败。

## 6.4 安全性/韧性/隐私保护

- 默认行为是读取框架配置、设备身份、Linux sysfs、CPU/NUMA 状态和必要的 Provider 查询结果，不修改内核、驱动、cgroup 或容器资源配置。
- Provider 使用的外部查询工具必须通过明确的可执行文件路径或受控 `PATH` 解析；不得把未校验的设备标识、BDF 或用户输入拼接为 shell 命令。
- 插件不采集模型输入、输出、请求内容或用户身份信息；日志只应记录设备映射、拓扑证据、决策来源和错误诊断所需的信息。
- 日志不得默认输出访问凭据、环境中的敏感变量或完整业务请求。外部框架和 Runtime 自身产生的日志仍按其安全策略管理。
- Provider、sysfs 和框架接口不可用时，系统通过明确的跳过、回退或严格失败保持可预测行为，不以未验证的设备顺序替代身份。
- 插件不依赖网络服务、远端数据库或在线控制面；本地依赖不可用时应在启动阶段给出可定位的错误信息。

## 6.5 易用性

- 默认配置应保持框架原有行为可用；用户只需在需要时选择 `auto`、`strict` 或指定 Provider。
- 配置项必须有明确的名称、取值范围、默认值和优先级，并区分“启用框架绑核”和“为框架补充 NUMA 结果”两个动作。
- `FORCE_GENERIC` 类诊断开关必须明确标记为验证用途，不得被解释为生产环境的通用启用开关。
- 日志应能让使用者区分插件未加载、Provider 未选中、拓扑发现失败、结果被框架原生配置覆盖和实际执行失败。
- 应提供不启动模型服务的只读拓扑/Provider 诊断路径，便于先验证设备映射和 BDF 拓扑。

## 6.6 可服务性

- 版本、Provider、模式、决策来源、逻辑设备、BDF、NUMA 节点和失败原因应通过结构化或稳定格式日志暴露。
- 诊断输出应包含足以复现判断的只读输入摘要，但不应包含模型请求或敏感环境变量。
- 代码应允许在不修改框架源码的情况下替换或新增 Provider，并允许对框架适配层单独进行契约测试。
- 兼容性问题应优先在插件加载阶段发现，报告框架版本、目标模块、函数签名或入口状态，而不是等到服务运行中才出现模糊错误。
- 真实服务验收应同时保存插件日志、框架执行日志、子进程 CPU affinity 和内存策略证据；“安装成功”或“拓扑 probe 成功”不能单独作为绑核生效证明。

## 6.7 兼容性

- 支持判断不能只依赖版本字符串，还必须检查目标框架模块、函数签名、插件加载时机、设备 visibility 语义和实际调用路径。
- vLLM 和 SGLang 的适配代码必须保持隔离；某一框架的接口变化不得通过修改通用拓扑分析器来补偿。
- 不同 GPU Runtime 的身份获取逻辑必须保持在独立 Provider 中；Provider 不得假设其它 Runtime 的设备节点、UUID 或枚举规则。
- 对未完成真实调用链验证的并行模式、服务启动器、框架本地构建和厂商后缀版本，只能标记为待验证或安全跳过，不得默认为兼容。
- 显式用户配置和框架原生有效结果具有更高优先级；插件不得覆盖它们以强制使用通用结果。

## 6.9 对操作系统的依赖（可选）

- 目标系统为 Linux，并需要可读的 PCI、NUMA 和 CPU sysfs，包括 PCI 设备父链、设备 `numa_node` 或等价 NUMA 证据、节点 CPU 列表及在线 CPU 列表。
- 插件运行时要求 Python `>=3.10`；Python 包的基础依赖以 `pyproject.toml` 为准，当前不把 GPU Runtime 或框架打包进插件依赖。
- 框架实际执行绑核时，依赖目标路径可用的 `libnuma`、`numactl` 或框架内部等价执行能力。不同框架和执行模式的必需项可能不同，不能仅凭插件安装成功推断这些组件存在。
- 容器或服务管理器提供的 cpuset、内存节点限制、权限和 seccomp 策略不得阻止框架读取拓扑或执行其绑定系统调用。插件不得扩大这些限制。
- 操作系统、内核、GPU 驱动和厂商 Runtime 的具体版本不作为插件单独保证；必须纳入目标平台的兼容性验证矩阵。

## 6.12 对外部系统的依赖（可选）

- **推理框架**：依赖目标 vLLM 或 SGLang 提供的插件入口、NUMA 配置语义和子进程启动路径。
- **GPU Runtime/驱动**：Provider 依赖 Runtime 或驱动导出的逻辑设备身份、UUID、BDF 映射表或其它明确设备身份接口。插件不自行推导缺失的硬件身份。
- **Linux 拓扑接口**：依赖 `/sys` 暴露的 PCIe、NUMA 和 CPU 信息；该接口不可用或证据矛盾时必须失败关闭或安全回退。
- **NUMA 执行组件**：实际 CPU/内存绑定由框架及其 `numactl`、`libnuma` 或进程内执行路径完成，插件只提供可消费的结果。
- **容器与服务启动器**：依赖服务进程能够加载插件包、继承必要环境变量并创建框架子进程。插件不负责修改容器镜像或服务编排配置。
- **外部查询工具**：Iluvatar 等 Provider 可能依赖目标环境提供的只读设备查询工具；工具缺失、输出格式变化或权限不足时，Provider 必须报告不可用而不是猜测映射。

## 6.13 其它：运行模式与验收边界

- 插件 `off` 表示不安装或不执行插件 Hook；它不等价于关闭框架自身的 NUMA/CPU 绑定。需要比较“绑核”和“不绑核”时，基线必须显式关闭框架绑定，并记录该配置。
- 插件 `auto`、`strict` 和诊断开关的行为必须在部署文档中分别说明；验证 generic 路径的开关不能直接当作生产部署开关。
- 功能验收至少应覆盖：Provider 设备映射、逻辑设备可见性重排、sysfs 拓扑分析、框架 Hook 调用、实际子进程 CPU affinity 和内存策略。性能验收另行记录测试条件和重复结果。
- 当前设计不承诺自动支持 Data Parallel、Ray 或其它未完成独立启动链验证的框架路径；新增路径必须先建立对应的适配和验证证据。

# 7 系统级规格设计

## 7.1.1 设计思路

系统采用“设备身份映射”和“Linux 拓扑分析”分离的设计。前者回答“当前框架逻辑设备对应哪一个 PCIe Endpoint”，后者回答“该 Endpoint 连接到哪个 NUMA 节点以及哪些 CPU”。两类问题使用不同的数据来源和校验规则，避免把厂商 Runtime 的设备枚举逻辑扩散到通用拓扑代码中。

一次完整处理按以下顺序进行：

1. 框架适配层读取当前框架版本、配置、进程类型和可见设备信息。
2. 适配层根据模式和配置判断是否允许插件参与；`off` 直接保持框架原有行为。
3. 对允许参与的请求，先保留用户显式配置，再判断框架原生 NUMA 结果是否完整有效。
4. 需要通用分析时，由 Provider 按当前逻辑设备顺序建立逻辑设备到 BDF 的映射。
5. 通用拓扑分析器从 BDF 开始读取 PCI sysfs，遍历父设备链并解析 NUMA 节点及 CPU 集合。
6. 决策层对结果进行设备数量、顺序、唯一性、拓扑证据和资源范围校验。
7. 校验通过后，以一次批量配置更新的方式把结果交给框架原有的执行路径；校验失败时不提交部分结果。

该设计不把“插件已加载”“Provider 已返回 BDF”或“拓扑分析成功”视为绑定已经生效。实际 CPU affinity 和内存策略仍由框架及其执行器完成，必须在真实服务验证中单独确认。

## 7.1.2 规格设计及分解

| 系统单元 | 输入 | 核心职责 | 输出或影响 | 设计约束 |
|---|---|---|---|---|
| 框架适配层 | 框架配置、版本、进程上下文、可见设备 | 识别 vLLM/SGLang 调用路径，检查兼容性，决定是否安装或调用 Hook | 有序逻辑设备上下文、适配层决策 | 不修改框架源码；不同框架的接口逻辑隔离 |
| Provider 注册与选择 | Provider 配置、Runtime 能力、逻辑设备列表 | 选择唯一适用的设备身份 Provider | Provider 实例及其映射结果 | 不使用未经证明的物理枚举顺序；不让多个 Provider 竞争提交 |
| 设备身份映射 | 逻辑设备 ID、visibility 信息、Runtime 查询结果 | 建立逻辑设备到物理设备及 BDF 的一一对应关系 | 规范化 BDF 列表 | 必须覆盖全部当前可见设备；身份不完整时失败关闭 |
| BDF 规范化 | 长/短格式 BDF、设备映射数据 | 统一 domain、bus、device、function 表示 | 可用于 sysfs 查找的规范化 BDF | 不改变 bus/device/function；格式异常不得静默修正为猜测值 |
| PCIe/NUMA 拓扑分析 | 规范化 BDF、PCI sysfs、NUMA 和 CPU sysfs | 遍历父设备链，解析 NUMA 证据并计算可用 CPU 集合 | 每个设备的 NUMA 节点、CPU 集合和证据 | 只使用可读的 Linux 拓扑信息；结果不得超出当前进程允许资源 |
| 决策与批量校验 | 显式配置、原生结果、通用分析结果、运行模式 | 按优先级选择来源，区分保留、回退、跳过和严格失败 | 可提交的批量 NUMA 配置或明确诊断 | 全量校验、原子提交；不提交部分成功结果 |
| 执行衔接层 | 已校验的 NUMA/CPU 配置、框架原有启动上下文 | 将结果传递给框架原有子进程启动和绑定路径 | 由框架执行 CPU/内存绑定 | 不在插件内重复实现 `numactl`、`libnuma` 或 `sched_setaffinity` |
| 诊断与可观测性 | 版本、Provider、设备、拓扑和决策状态 | 输出可定位的结构化或稳定格式信息 | 日志、只读 probe 和错误码 | 不记录模型请求、凭据或无关敏感变量 |

新增设备 Runtime 时，只需在 Provider 边界实现该 Runtime 的身份获取、可见性解释和 BDF 映射，并复用已有的 BDF 规范化、拓扑分析、校验和执行衔接逻辑。新增框架时，应在对应 Adapter 中实现入口、版本和调用签名适配，不应通过修改通用拓扑分析器来解决框架接口差异。

# 9 产品系统定义

## 9.1 系统定位

鲲鹏 GPU 亲和性插件是部署在推理框架 Python 运行环境中的独立扩展组件。它位于框架的 NUMA 查询或子进程配置入口附近，为框架补充 GPU 到 NUMA/CPU 亲和性所需的设备身份和拓扑信息。插件不是 GPU 驱动、GPU Runtime、推理框架或独立的 CPU 绑定服务。

## 9.2 系统组成

产品系统由以下逻辑部分组成：

- **框架适配组件**：分别承载 vLLM 和 SGLang 的入口加载、版本检查、调用签名适配和配置语义处理。
- **Provider 组件**：按设备 Runtime 提供逻辑设备到 BDF 的映射。当前设计包含通用框架设备映射路径、Iluvatar Runtime 路径和 Ascend sysfs 路径；具体 Provider 是否可用由目标环境能力决定。
- **通用拓扑组件**：读取 Linux PCI、NUMA 和 CPU sysfs，完成 BDF 规范化、PCIe 父链遍历、NUMA 解析和 CPU 集合计算。
- **决策与契约组件**：处理显式配置、原生结果、通用回退、运行模式、结果校验和批量提交。
- **诊断组件**：提供只读拓扑/Provider 检查和稳定日志，帮助区分插件决策成功与框架执行生效。

## 9.3 系统边界

插件内部的输入是框架启动上下文、当前可见逻辑设备、设备 Runtime 身份信息和 Linux sysfs 拓扑信息；插件内部的输出是经校验的 NUMA 或 CPU 亲和性配置，以及诊断信息。

插件外部但属于运行前提的系统包括：

- vLLM 或 SGLang 的插件加载和子进程启动机制；
- GPU 驱动、GPU Runtime 及其设备身份接口；
- Linux 内核提供的 PCI、NUMA、CPU sysfs；
- 框架实际使用的 `libnuma`、`numactl` 或等价进程内绑定能力；
- 容器或服务管理器提供的 Python 环境、文件可见性、CPU/内存权限和子进程能力。

这些外部系统不由插件安装、升级或替换。外部系统缺失时，插件只能按运行模式执行安全跳过、回退或严格失败。

## 9.4 系统数据流

```text
框架启动上下文
  -> Adapter 识别与模式判断
  -> 当前可见逻辑设备
  -> Provider 获取 BDF
  -> Linux sysfs 拓扑分析
  -> NUMA/CPU 结果批量校验
  -> 框架原有执行器
  -> 子进程 affinity 与内存策略
```

其中最后一条执行链不由插件实现。插件对最后一步的产品责任是提供正确、完整、可消费的配置并输出可核验的决策证据；绑定系统调用是否成功由框架执行器和操作系统权限共同决定。

# 10 词汇表

| 名称 | 描述 |
|---|---|
| 鲲鹏 GPU 亲和性插件 | 本文描述的独立 Python 插件，用于为推理框架补充 GPU 到 CPU/NUMA 的亲和性信息。 |
| 逻辑 GPU | 框架进程在当前可见性配置下使用的设备编号；它不一定等于整机物理设备枚举编号。 |
| 物理设备 | Runtime 或驱动所识别的实际加速设备实体。 |
| BDF | PCIe 设备地址的 Bus、Device、Function 组合，本文用它作为 PCIe Endpoint 的身份输入。 |
| PCIe Endpoint | 设备连接到 PCIe 拓扑中的端点设备，通常是 GPU 或其它加速器对应的 PCI 设备。 |
| PCIe 父链 | 从 Endpoint 沿 Linux sysfs 的设备关系向上经过的桥、Root Port 或 Host Bridge 链路。 |
| NUMA 节点 | 具有相对局部内存访问关系的 CPU 和内存资源集合。 |
| NUMA 亲和性 | 进程或线程优先使用指定 NUMA 节点 CPU/内存资源的关系。 |
| CPU 亲和性 | 进程或线程可运行 CPU 集合的约束。 |
| Runtime Provider | 负责从特定设备 Runtime 或驱动接口获取设备身份并映射到 BDF 的插件组件。 |
| Adapter | 面向具体推理框架的适配组件，负责入口、版本、签名和配置语义适配。 |
| Provider Registry | 管理 Provider 注册、能力匹配和选择的组件。 |
| visibility | 当前进程能够看到和使用的设备集合及其逻辑顺序。 |
| 原生查询 | 推理框架自身提供的 GPU 到 NUMA 或 PCIe 信息查询路径。 |
| 通用回退 | 原生查询缺失或返回可恢复空结果时，使用 Provider 和 Linux sysfs 完成拓扑分析的路径。 |
| 显式配置 | 用户或服务启动参数直接指定的 NUMA、CPU 或绑定配置，优先级高于自动发现。 |
| 批量校验 | 对当前全部可见设备的映射和拓扑结果进行完整性、一致性和可提交性检查。 |
| 原子提交 | 只有整批结果全部通过校验时才更新框架配置，不提交部分结果。 |
| `off` 模式 | 不安装或不执行插件 Hook，保持插件不参与当前框架流程。 |
| `auto` 模式 | 允许插件自动发现；可恢复失败时记录原因并尽量保持框架原有启动行为。 |
| `strict` 模式 | 要求插件发现和校验成功；不满足条件时明确报告错误。 |
| `FORCE_GENERIC` | 用于隔离验证通用发现路径的诊断开关，不负责选择 Provider，也不负责开启框架绑核。 |
| 绑定执行器 | 框架或其依赖组件中实际执行 CPU affinity、NUMA memory policy 等系统操作的部分。 |
| 设备可见性重排 | 通过进程可见设备配置改变逻辑设备顺序，同时保持逻辑设备到物理身份的正确映射。 |
| 失败关闭 | 无法证明身份、拓扑或权限条件时，不生成猜测性配置；根据模式安全跳过或严格失败。 |
