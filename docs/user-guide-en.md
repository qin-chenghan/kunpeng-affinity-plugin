# Kunpeng GPU Affinity Plugin User Guide

## 1. About This Guide

This guide is intended for installation, operations, and validation personnel who deploy inference services with vLLM or SGLang. It describes how to install, configure, enable, verify, troubleshoot, and uninstall the Kunpeng GPU Affinity Plugin.

When a framework has no usable GPU-to-NUMA result, the plugin supplements the resolution from a logical GPU to its PCIe BDF, NUMA node, and CPU set, then hands the validated result to the framework's existing binding executor. The plugin does not install a GPU driver or inference framework, change operating-system resource limits, or replace framework CPU and memory binding.

## 2. Supported Scope

### 2.1 Inference Frameworks

| Framework | Supported base versions | Notes |
|---|---|---|
| vLLM | `0.23.0`, `0.25.1`, `0.26.0` | Vendor-local suffixes of these base versions are accepted. Other versions are not implicitly compatible. |
| SGLang | `0.5.18`, `0.5.17.dev386+gc5bd3d7dc` | Ordinary Engine subprocess path. Data Parallel and Ray paths are outside the current support scope. |

A version match only means that the plugin permits Hook installation. It does not prove every hardware, Runtime, and launch path. Complete the real-service verification in Section 8 before deployment.

### 2.2 Device Mapping Providers

| Provider | Applicable condition |
|---|---|
| `auto` | Selects the only Provider that supports the complete current device batch. Recommended for normal deployment. |
| `vllm-platform-pci` | The vLLM platform interface returns a complete BDF mapping for every visible logical device. |
| `sglang-runtime-pci` | The Runtime used by SGLang returns a direct BDF or a device identity that can be joined to one. |
| `iluvatar-runtime-pci` | The Runtime returns logical-device UUIDs and `ixsmi` returns a UUID-to-BDF inventory. |
| `ascend-sysfs-pci` | An ordered `ASCEND_RT_VISIBLE_DEVICES` is set and the Ascend driver sysfs mapping is readable. |

## 3. Prerequisites

Perform all checks and installation steps inside the container or Python environment that starts the inference service, using the same user, interpreter, device visibility, and resource limits as the service.

### 3.1 Common Requirements

- A Linux system with multiple NUMA nodes. A single-node system has no automatic NUMA optimization target.
- Python 3.10 or later.
- Python package `packaging>=21`.
- A supported vLLM or SGLang installation, plus the device driver, Runtime, and framework extension required by that accelerator.
- Readable PCI, NUMA, and CPU sysfs information.
- Loadable and usable `libnuma.so` or `libnuma.so.1`.
- vLLM and the SGLang V2 path require `numactl` on `PATH`. SGLang V1 does not require the command but still requires libnuma and memory-policy permissions.
- The container or service process must allow the target CPUs, memory nodes, related system calls, and framework subprocess creation.
- The vLLM multiprocessing path uses `spawn`. SGLang V2 also requires a writable and executable temporary directory.

The plugin does not install these dependencies.

### 3.2 Provider Requirements

**Iluvatar**

- The framework or Torch Runtime can obtain the UUID of the current logical device.
- `ixsmi` is executable and its driver and Runtime libraries are available in the service environment.
- The `ixsmi` query returns device index, UUID, and PCI BDF. The plugin joins devices by UUID and never infers logical identity from physical enumeration order.

**Ascend**

- `ASCEND_RT_VISIBLE_DEVICES` explicitly supplies the ordered visible physical-device IDs.
- The driver-exported `devdrv_sysfs_bdf_to_devid` sysfs attribute is readable and covers every visible device.
- The Provider does not depend on `npu-smi`, DCMI, or `/dev/davinciN` suffixes for identity inference.

### 3.3 Prerequisite Checks

```bash
<python> --version
<python> -m pip check
command -v numactl
numactl --hardware
numactl --show
```

The `numactl` checks are optional for SGLang V1. These results establish only the current process prerequisites; they do not replace binding verification on the real EngineCore or Worker process.

## 4. Installation and Removal

### 4.1 Installation

Install the plugin into the same Python environment that starts the inference framework. For a released wheel artifact, run:

```bash
<python> -m pip install <wheel-file>
```

To install from delivered source, run this command from the source root:

```bash
<python> -m pip install --no-deps --no-build-isolation .
```

For source development or on-site debugging that requires an editable installation, run:

```bash
<python> -m pip install --no-deps --no-build-isolation -e .
```

`--no-deps` does not install missing dependencies. Verify Python, `packaging`, the framework, and system NUMA components before installation.

### 4.2 Verify Package Installation

```bash
<python> - <<'PY'
from importlib.metadata import entry_points, version

print("plugin_version:", version("kunpeng-affinity-plugin"))
for group in ("vllm.general_plugins", "sglang.srt.plugins"):
    matches = [ep.value for ep in entry_points(group=group) if ep.name == "kunpeng_affinity"]
    print(group, matches)
PY
```

Both entry-point groups should display the plugin registration function. This check proves only that package metadata is visible. It does not prove that a framework executed the plugin entry point or applied binding.

### 4.3 Removal

After stopping new service processes that use the plugin, run:

```bash
<python> -m pip uninstall kunpeng-affinity-plugin
```

The plugin does not modify framework source code. Newly started framework processes return to their behavior without the plugin after removal.

## 5. Configuration

### 5.1 Common Configuration

| Environment variable | Values | Default | Description |
|---|---|---|---|
| `KUNPENG_AFFINITY_MODE` | `off`, `auto`, `strict` | `auto` | Disables the plugin, enables automatic fallback, or requests strict failure. |
| `KUNPENG_AFFINITY_PROVIDER` | `auto` or a registered Provider name | `auto` | Selects the device mapping source. |
| `KUNPENG_AFFINITY_CPU_POLICY` | `node`, `exact` | `node` | Current framework adapters support only `node`. Do not use `exact` in production configuration. |
| `KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL` | `error`, `summary`, `detail` | `summary` | Controls plugin diagnostic detail. |
| `KUNPENG_AFFINITY_CONFIG` | UTF-8 JSON file | unset | Selects an optional configuration file. Environment variables override matching file fields. |
| `KUNPENG_AFFINITY_IXSMI` | executable path | `ixsmi` on `PATH` | Selects the Iluvatar query utility. |

Recommended normal production configuration:

```bash
export KUNPENG_AFFINITY_MODE=auto
export KUNPENG_AFFINITY_PROVIDER=auto
export KUNPENG_AFFINITY_CPU_POLICY=node
export KUNPENG_AFFINITY_DIAGNOSTIC_LEVEL=summary
```

### 5.2 JSON Configuration File

The configuration root must be a JSON object and may contain only these fields:

```json
{
  "mode": "auto",
  "provider": "auto",
  "cpu_policy": "node",
  "diagnostic_level": "summary"
}
```

Unknown fields, duplicate fields, invalid values, non-UTF-8 content, or a non-object root cause configuration failure. When `KUNPENG_AFFINITY_MODE=off` is set, the plugin exits before reading the optional configuration file.

### 5.3 Operating Modes

| Mode | Behavior |
|---|---|
| `off` | Does not install or execute the plugin Hook. It does not disable framework-native binding. |
| `auto` | Preserves explicit configuration and valid native results first, and permits generic discovery. On a plugin discovery failure, it records the reason and attempts to preserve normal framework startup. |
| `strict` | Reports plugin discovery or validation failures. Framework configuration and binding-executor errors remain framework errors. |

## 6. Using the Plugin with vLLM

### 6.1 Start a Service

vLLM must explicitly enable its NUMA binding switch. Do not provide explicit `numa_bind_nodes` at the same time, because the plugin preserves that explicit configuration instead of running automatic discovery.

```bash
export KUNPENG_AFFINITY_MODE=auto
export KUNPENG_AFFINITY_PROVIDER=auto
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve <model> --numa-bind <other-vllm-options>
```

If the deployment already uses a vLLM plugin filter, it must allow `kunpeng_affinity`. Installing the package does not add `--numa-bind` automatically.

### 6.2 vLLM Decision Order

```text
explicit user NUMA nodes
  -> valid vLLM native result
  -> Provider BDF mapping and generic Linux sysfs analysis
  -> original vLLM numactl execution path
```

The plugin automatically fills only the missing NUMA node list and preserves a user-provided CPU list.

### 6.3 Forced Generic-Path Diagnostics

`KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1` is only for isolated validation. It bypasses the native vLLM NUMA query and forces validation of the Provider and Linux topology path. It does not enable `--numa-bind`, does not select a Provider, and reports generic discovery failure even in `auto` mode.

Leave it unset for normal deployment:

```bash
unset KUNPENG_AFFINITY_VLLM_FORCE_GENERIC
```

## 7. Using the Plugin with SGLang

### 7.1 Start a Service

```bash
export KUNPENG_AFFINITY_MODE=auto
export KUNPENG_AFFINITY_PROVIDER=auto
export SGLANG_PLUGINS=kunpeng_affinity
export SGLANG_AUTO_NUMA_BIND=1
export SGLANG_NUMA_BIND_V2=1

sglang serve <model> <other-sglang-options>
```

`SGLANG_NUMA_BIND_V2=1` uses the external SGLang `numactl` path. Set it to `0` only when explicitly selecting SGLang V1. V1 uses in-process libnuma and does not require the `numactl` command, but it still requires libnuma and memory-policy permissions. V1 and V2 are explicit execution-mode choices; the plugin does not switch between them automatically.

Generic fallback is not used in these conditions:

- `SGLANG_AUTO_NUMA_BIND=0`;
- an explicit NUMA node is already configured;
- `SGLANG_SET_CPU_AFFINITY=1` is enabled, because it would overwrite NUMA-derived CPU affinity;
- an unsupported Data Parallel or Ray launch path is used.

### 7.2 SGLang Decision Order

```text
explicit user NUMA node
  -> SGLang native query result
  -> Provider BDF mapping and generic Linux sysfs analysis
  -> original SGLang binding execution path
```

## 8. Verify Effective Binding

Verify the following items in order before production deployment. Package installation alone does not prove that the plugin is effective.

1. **Package visibility**: the service interpreter can import the plugin and see the appropriate entry point.
2. **Plugin loading**: service logs contain plugin Hook installation or registration output.
3. **Generic-path selection**: when generic fallback is required, logs contain the Provider, logical device, BDF, and selected NUMA node.
4. **Framework binding execution**: inspect `Cpus_allowed_list` on the actual EngineCore, Worker, or Engine subprocess.
5. **Memory-policy execution**: use an approved target-environment tool or process interface to confirm the applied memory policy. Merely inspecting allowed memory nodes does not prove policy application.
6. **Service availability**: run a health check and a fixed inference request to confirm normal startup and responses.

In `auto` mode, the plugin may safely skip an unsupported or unavailable path. Therefore, successful service startup does not imply that plugin-assisted binding occurred.

## 9. Troubleshooting

### 9.1 No Plugin-Loading Message

- Confirm that the plugin is installed in the Python environment used by the service.
- Confirm that the framework version is supported.
- Check that framework plugin filtering permits `kunpeng_affinity`.
- Confirm that `KUNPENG_AFFINITY_MODE` is not `off`.

### 9.2 Plugin Loaded but No Generic NUMA Selection Message

- vLLM requires `--numa-bind` and must not already have explicit NUMA nodes.
- SGLang requires `SGLANG_AUTO_NUMA_BIND=1`.
- A valid native framework result is preserved; this is expected behavior.
- Check whether the launch path is an unsupported DP, Ray, or independent path.

### 9.3 Binding Executor Is Missing

- vLLM and SGLang V2 require `numactl` on the service process `PATH`.
- SGLang V1 does not require the command, but still requires loadable libnuma and permission to use memory policies.
- A host-installed tool does not compensate for a missing tool or library inside a container.

### 9.4 Provider Cannot Map Devices

- Iluvatar: confirm that logical-device UUIDs are available and `ixsmi` is executable and returns UUID/BDF data. Set `KUNPENG_AFFINITY_IXSMI` when necessary.
- Ascend: confirm the order of `ASCEND_RT_VISIBLE_DEVICES` and verify that the driver sysfs mapping is readable and covers every visible device.
- Never replace logical-device mapping with physical enumeration order.

### 9.5 Memory Policy or Affinity Is Denied

- Check container cpuset and allowed memory nodes.
- Check seccomp policy, system-call permissions, and temporary-directory execution permissions.
- Do not broaden container privileges or resources to hide an incorrect mapping. Grant only the permissions required by the deployment security policy.

## 10. Usage Boundaries

- The plugin performs topology discovery only during framework startup or subprocess configuration, not on the per-request inference path.
- Plugin `off` is not the same as disabling all framework binding. A performance baseline must explicitly disable framework-native binding as well.
- Plugin logs prove loading, discovery, and decision behavior. Actual binding must be verified on the target subprocess.
- The plugin does not promise a fixed performance gain. Performance conclusions require repeated A/B testing with the same model, parallelism, requests, framework options, and system load.
- Framework versions, device Runtimes, and launch paths outside the support scope require independent compatibility validation before use.
