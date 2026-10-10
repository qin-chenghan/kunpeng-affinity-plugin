# SGLang Ascend TP=4 阶段性验证记录

## 1. 记录范围

本文记录 SGLang 普通 Engine、Ascend 多卡和 TP=4 场景截至 2026-10-10 已获得的证据。当前记录用于追踪验证进展，不作为最终性能报告。

## 2. 被测对象

| 项目 | 当前记录 |
|---|---|
| CPU 与加速器 | 鲲鹏 920B、Ascend 910B 系列 |
| 框架 | 已观察的 SGLang 0.5.17.dev386+gc5bd3d7dc 兼容构建 |
| 模型 | Qwen3-32B-W8A8 |
| 并行方式 | TP=4，使用可见设备 `0,1,2,3` |
| 插件核心修复 | `9eab78f` |
| 当前脚本修复 | `d2d27eb` |
| 当前源码测试 | 175 项通过 |

不记录服务器 IP、登录信息、宿主机绝对路径或容器外部业务配置。

## 3. 已获得的功能证据

| 检查项 | 当前结果 | 证据边界 |
|---|---|---|
| 服务启动 | 通过 | SGLang TP=4 服务可以完成启动 |
| 推理请求 | 通过 | off/on 两种启动方式均能完成请求 |
| 插件注册与 Hook | 已观察到相关日志 | 仍需在最终报告中保留完整注册、Applied hook 和决策日志 |
| Ascend 设备识别 | 通过 | 不再因 `torch.cuda.device_count()==0` 跳过；从 `ASCEND_RT_VISIBLE_DEVICES` 构造批次 |
| Provider 映射 | 已有独立真机证据 | 可见设备 0、1、2、3 分别映射到 BDF，并得到 NUMA 6、6、4、4；该结果本身不等同于服务进程已完成绑定 |
| 服务进程 CPU affinity | 待补 | 尚未形成逐 PID、逐 rank 的 `Cpus_allowed_list` 记录 |
| 服务进程 memory policy | 待补 | 不能用 `Mems_allowed_list` 代替实际 memory policy |

## 4. 初步性能数据

以下数据由旧版脚本采集，每个模式只有 3 个串行请求，因此只记录现象，不形成正式收益结论。

| 输入/输出 | 指标 | off | on | 观察 |
|---|---|---:|---:|---|
| 128/128 | TTFT mean | 187.0 ms | 185.5 ms | 基本持平，on 略低 |
| 128/128 | TPOT mean | 20.44 ms | 20.26 ms | on 约降低 0.9% |
| 128/128 | E2E mean | 2.784 s | 2.759 s | on 约降低 0.9% |
| 128/128 | throughput | 45.98 tok/s | 46.39 tok/s | on 约提高 0.9% |
| 2048/2048 | TTFT p50 | 222.8 ms | 201.3 ms | on 约降低 9.7% |
| 2048/2048 | TPOT mean | 22.57 ms | 22.20 ms | on 约降低 1.6% |
| 2048/2048 | E2E mean | 47.619 s | 45.626 s | 受 off 冷启动样本影响 |
| 2048/2048 | throughput | 42.96 tok/s | 44.84 tok/s | 受 off 冷启动样本影响 |

2048/2048 的 off 第一条请求 TTFT 约为 4.013 秒，后两条约为 201 毫秒和 223 毫秒；on 三条约为 309 毫秒、201 毫秒和 196 毫秒。off 均值不能用于归因，稳态 TPOT 和吞吐方向可以作为后续重测的参考。

## 5. 脚本修正

提交 `d2d27eb` 对机器专用脚本做了小范围修正：

1. 实际测量默认使用 32 请求、并发 8，与脚本显示配置一致；
2. off 模式显式设置 `KUNPENG_AFFINITY_MODE=off` 和 `SGLANG_AUTO_NUMA_BIND=0`；
3. 每个 case 正式采样前执行一次同输入长度、短输出预热，并清理 SGLang cache；
4. 删除旧 `latency.json`，避免失败时误读历史结果；
5. 只有全部请求成功才判定 case 成功；
6. 缺少结果文件时明确失败。

该提交通过 shell 语法检查、diff 检查和 175 项源码单元测试。目标环境尚未回传新版性能结果。

## 6. 当前结论

SGLang Ascend TP=4 已经进入真实服务验证，不再属于“仅源码实现”。现有证据支持以下结论：

- 插件可以在目标 SGLang 版本中注册并进入 Ascend 通用设备识别路径；
- TP=4 服务和推理请求可以正常运行；
- 初步性能结果未显示系统性劣化，TPOT 和吞吐方向为正。

当前不能宣称正式性能收益，也不能仅凭 Provider 日志和性能变化认定每个服务进程的 CPU affinity 与 memory policy 已正确生效。

## 7. 验收待办

1. 使用 `d2d27eb` 的脚本重跑 32 请求、并发 8 的 off/on 对比；
2. 保留 on 模式的插件注册、SGLang Applied hook 和每个 GPU 的 NUMA 选择日志；
3. 建立 PID、进程角色、rank、逻辑设备、BDF、NUMA node、`Cpus_allowed_list` 的对应表；
4. 独立记录每个目标进程的实际 NUMA memory policy；
5. 核对 off 模式没有安装插件 Hook 或提交 generic NUMA 结果；
6. 将功能证据、白盒绑定证据和新版性能数据合并为正式自验证报告。
