# 通用 GPU 亲和性探测 Demo

这个 Demo 是只读的跨框架验证工具，用于在安装或接入 vLLM/SGLang 插件前检查一台 Linux 机器是否能够完成：

```text
设备输入或候选发现
  -> PCI BDF
  -> Linux PCIe 父路径
  -> NUMA 归属
  -> node CPUs ∩ online CPUs ∩ allowed CPUs
```

它不依赖 vLLM、SGLang、NVML、`nvidia-smi` 或其他厂商工具，也不会调用 `sched_setaffinity`、`numactl` 或启动 GPU 工作负载。

## 运行

在项目根目录执行：

```bash
./demo/probe-generic-affinity.sh
```

不提供 BDF 时，脚本会按 PCI base class `03`（显示控制器）和 `12`（处理加速器）发现候选设备。这个顺序只是 Linux PCI sysfs 枚举顺序，不能当作框架逻辑 GPU 顺序。

对可信的设备顺序，显式提供 BDF：

```bash
./demo/probe-generic-affinity.sh \
  --bdf 0000:41:00.0 \
  --bdf 0000:81:00.0
```

也可以使用环境变量：

```bash
AFFINITY_BDFS=0000:41:00.0,0000:81:00.0 \
  ./demo/probe-generic-affinity.sh
```

输出 JSON 便于保存到不同机器比较：

```bash
./demo/probe-generic-affinity.sh --json > affinity-result.json
```

自动发现时可按厂商或驱动过滤：

```bash
./demo/probe-generic-affinity.sh --vendor-id 0x1e3e --driver iluvatar
```

测试或受限环境可以覆盖 CPU 允许集合：

```bash
./demo/probe-generic-affinity.sh --bdf 0000:41:00.0 --allowed-cpus 0-31,64-95
```

## 结果解释

- `success`：BDF、PCIe 父路径、NUMA 证据和非空目标 CPU 集均已确认。
- `partial`：PCIe 路径可见，但无法证明唯一 NUMA 归属。
- `failed`：BDF、路径、NUMA 证据或最终 CPU 集存在错误或冲突。

脚本只输出每个设备的 NUMA node 和 CPU 集，不生成 vLLM 或 SGLang 参数。因为通用脚本没有框架 Provider，自动 PCI 枚举也不等同于逻辑 GPU 映射；只有显式 BDF 顺序或后续 Runtime Provider 才能完成这一步。

返回码为 `0` 表示所有设备均为 `success`，`2` 表示存在不完整结果，`3` 表示没有设备候选或参数无效。
