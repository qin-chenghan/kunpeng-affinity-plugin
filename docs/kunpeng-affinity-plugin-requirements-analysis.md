# 鲲鹏 GPU 亲和性插件需求分析说明书

# 1 需求分析说明书

## 1.4 Keywords 关键词

| English | 中文 |
|---|---|
| GPU CPU affinity | GPU CPU 亲和性 |
| NUMA-aware inference | NUMA 感知推理 |
| Device identity mapping | 设备身份映射 |
| PCIe topology | PCIe 拓扑 |
| PCI Bus-Device-Function | PCI 总线-设备-功能地址 |
| Runtime Provider | 运行时设备映射 Provider |
| Linux sysfs | Linux sysfs 文件系统 |
| vLLM and SGLang integration | vLLM 与 SGLang 集成 |

## 1.5 Abstract 摘要

### 中文摘要

在多 NUMA 节点的鲲鹏推理服务器中，GPU 与 CPU、内存之间的拓扑距离会影响主机侧调度、数据传输和内存访问效率。vLLM、SGLang 等推理框架具备进程绑核或 NUMA 绑定能力，但部分 GPU Runtime、框架后端或特定版本无法为当前可见的逻辑 GPU 提供完整、可信的 NUMA 拓扑信息，导致框架已有的自动绑定能力无法正确使用。

鲲鹏 GPU 亲和性插件旨在补充这一信息缺口。系统需要在不修改推理框架源码、不替代 GPU 驱动和不突破容器资源边界的前提下，将框架可见的逻辑设备可靠映射为 PCIe BDF，并利用 Linux 提供的 PCIe、NUMA 和 CPU 信息生成框架可消费的亲和性配置。系统还需要尊重用户显式配置和框架原生有效结果，在信息不足或结果冲突时禁止猜测性绑定。

本说明书定义该产品的业务背景、目标、范围、参与对象以及功能和非功能需求边界，为系统设计、开发、测试、验收和交付提供统一依据。插件是否成功发现拓扑与框架是否实际完成 CPU/内存绑定属于不同验收环节，均需保留可核验的证据。

### English Abstract

On Kunpeng inference servers with multiple NUMA nodes, the topology distance between GPUs, CPUs, and memory can affect host-side scheduling, data transfer, and memory access efficiency. Inference frameworks such as vLLM and SGLang provide process affinity or NUMA binding capabilities, but some GPU runtimes, framework backends, or specific versions cannot provide complete and trustworthy NUMA topology information for the logical GPUs visible to the process. As a result, the framework's existing automatic binding capability cannot always be used correctly.

The Kunpeng GPU Affinity Plugin is intended to fill this information gap. Without modifying inference framework source code, replacing GPU drivers, or exceeding container resource boundaries, the system shall reliably map framework-visible logical devices to PCIe BDF addresses and use Linux PCIe, NUMA, and CPU information to produce affinity configuration consumable by the framework. It shall preserve explicit user configuration and valid framework-native results, and it shall never guess a binding target when identity or topology evidence is incomplete or inconsistent.

This document defines the product background, objectives, scope, stakeholders, and functional and non-functional requirement boundaries. It provides a common basis for system design, implementation, testing, acceptance, and delivery. Successful topology discovery and successful CPU or memory binding by the framework are separate acceptance concerns and both require verifiable evidence.

## 1.6 List of Abbreviations 缩略语清单

| 缩略语 | English full name | 中文解释 |
|---|---|---|
| API | Application Programming Interface | 应用程序编程接口。 |
| BDF | Bus-Device-Function | PCI 设备的总线、设备和功能地址。 |
| CPU | Central Processing Unit | 中央处理器。 |
| DP | Data Parallelism | 数据并行。 |
| GPU | Graphics Processing Unit | 图形处理器；本文泛指用于推理计算的 GPU 或兼容加速设备。 |
| NUMA | Non-Uniform Memory Access | 非一致性内存访问架构。 |
| PCI | Peripheral Component Interconnect | 外设部件互连总线标准。 |
| PCIe | Peripheral Component Interconnect Express | PCI 高速串行总线及其设备互连结构。 |
| SPI | Service Provider Interface | 服务提供者接口；本文指设备映射 Provider 遵循的扩展边界。 |
| TP | Tensor Parallelism | 张量并行。 |
| UUID | Universally Unique Identifier | 通用唯一标识符；在支持的 Runtime 中用于稳定识别设备。 |
| sysfs | Linux system filesystem | Linux 内核向用户空间提供设备和系统信息的虚拟文件系统。 |

## 1.7 前言

本文档面向产品负责人、架构师、开发人员、测试人员、交付人员及目标环境运维人员。文档从使用场景和外部约束出发描述产品必须具备的能力及验收边界，不规定所有内部类、函数或文件组织；具体实现方案由系统设计说明书定义。

本文档中的“GPU”在不引起歧义时也包括由目标推理框架作为计算设备管理的兼容加速器。“绑核”包括 CPU affinity 配置；当目标框架支持时，还包括与 NUMA 节点相关的内存策略。插件负责发现、校验和提交亲和性配置，实际绑定动作由推理框架及其执行组件完成。

需求表述应区分以下三类状态：

- **产品需求**：系统为了满足目标场景必须具备的能力。
- **实现状态**：当前代码已经实现并通过静态或单元测试验证的能力。
- **环境验证状态**：在指定硬件、Runtime、框架版本和启动方式下完成的真实运行验证。

除非明确给出验证环境和证据，本文中的需求不得被解释为所有硬件、框架版本和并行模式均已完成兼容性验证；性能测试结果也不得被推广为无条件收益承诺。

# 2 简介

多卡推理服务通常由多个进程或线程分别使用不同的逻辑 GPU。对于多 NUMA 节点服务器，如果这些进程使用了远离目标 GPU 的 CPU 或内存资源，可能产生额外的跨 NUMA 访问。推理框架能够执行 CPU 和内存绑定，但自动选择绑定目标需要一条完整且可靠的信息链：当前逻辑 GPU、设备物理身份、PCIe BDF、Linux PCIe 路径、NUMA 节点以及进程实际允许使用的 CPU 和内存资源。

不同 GPU Runtime 暴露设备身份的方式并不一致；逻辑设备编号还可能因可见性配置发生重排。Linux 可以提供 PCIe 和 NUMA 拓扑，但仅凭物理 PCI 设备列表无法判断框架进程中的逻辑设备对应哪一个 BDF。因此，需要一个位于推理框架与 Linux 拓扑信息之间的独立插件，补全逻辑设备身份映射并复用框架已有的绑定执行能力。

## 2.1 目的

本文档用于：

- 说明鲲鹏 GPU 亲和性插件要解决的问题、目标用户和典型使用环境；
- 定义产品功能范围、外部接口边界、运行约束和失败处理要求；
- 明确插件发现亲和性配置与框架实际执行绑定之间的责任划分；
- 为系统设计、开发实现、测试用例、兼容性验证和交付验收提供可追溯的需求基线；
- 防止将实验性开关、特定机器结论或单次性能数据误解为通用产品能力。

## 2.2 范围

本产品覆盖以下范围：

- 以独立 Python 插件形式接入受支持的 vLLM 和 SGLang 版本；
- 识别框架当前进程可见的逻辑设备及其顺序；
- 通过框架原生接口或设备 Runtime Provider 获取逻辑设备对应的 PCIe BDF；
- 基于 Linux sysfs 读取 PCIe、NUMA 和 CPU 拓扑信息；
- 在进程允许的 CPU 和内存资源范围内生成可供框架使用的亲和性配置；
- 保留用户显式配置和有效的框架原生结果，在允许的条件下执行通用拓扑回退；
- 对多设备映射和拓扑结果进行完整性、一致性与可提交性校验；
- 提供自动、严格和关闭模式，以及必要的日志和只读诊断能力；
- 支持通过独立 Provider 扩展不同 GPU Runtime 的设备身份获取方式。

本产品不覆盖以下范围：

- 安装、升级或配置 GPU 驱动、GPU Runtime、推理框架、`libnuma`、`numactl` 或操作系统组件；
- 修改 Linux 内核、调度器、cgroup、容器资源限制或主机 NUMA 配置；
- 替代推理框架已有的子进程启动、CPU affinity 或内存策略执行器；
- 在无法建立逻辑设备到唯一 BDF 映射时根据物理枚举顺序猜测设备身份；
- 保证 CPU 核独占、实时调度、业务负载隔离或消除所有跨 NUMA 访问；
- 对未列入兼容性矩阵的框架版本、设备 Runtime、Data Parallel、Ray 或其它启动路径作默认支持承诺；
- 承诺固定的 TTFT、TPOT、吞吐或端到端时延提升。
