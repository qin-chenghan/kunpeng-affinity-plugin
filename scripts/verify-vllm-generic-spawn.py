#!/usr/bin/env python3
"""Verify the forced generic path and vLLM's real numactl spawn wrapper."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
from types import SimpleNamespace

from kunpeng_affinity.topology.cpulist import format_cpulist, parse_cpulist


def _status_value(name: str) -> str:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            key, separator, value = line.partition(":")
            if separator and key == name:
                return value.strip()
    raise RuntimeError(f"{name} is missing from /proc/self/status")


def _numactl_policy() -> dict[str, str]:
    result = subprocess.run(
        ["numactl", "--show"],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"policy", "cpubind", "nodebind", "membind"}:
            fields[key] = value.strip()
    return fields


def _stop_spawn_resource_tracker() -> None:
    """Reap the diagnostic's tracker when the container has no init process."""
    from multiprocessing import resource_tracker

    stop = getattr(resource_tracker._resource_tracker, "_stop", None)
    if callable(stop):
        stop()


def _child(send_connection) -> None:
    send_connection.send(
        {
            "pid": os.getpid(),
            "cpus_allowed_list": _status_value("Cpus_allowed_list"),
            "mems_allowed_list": _status_value("Mems_allowed_list"),
            "numactl": _numactl_policy(),
        }
    )
    send_connection.close()


def _config():
    parallel_config = SimpleNamespace(
        numa_bind=True,
        numa_bind_nodes=None,
        numa_bind_cpus=None,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    return SimpleNamespace(parallel_config=parallel_config)


def main() -> int:
    verification_path = os.environ.get(
        "KUNPENG_AFFINITY_VERIFY_PATH",
        "forced-generic",
    )
    if verification_path not in {"forced-generic", "auto-fallback"}:
        raise RuntimeError(
            "KUNPENG_AFFINITY_VERIFY_PATH must be forced-generic or auto-fallback"
        )
    os.environ["KUNPENG_AFFINITY_MODE"] = "auto"
    if verification_path == "forced-generic":
        os.environ["KUNPENG_AFFINITY_VLLM_FORCE_GENERIC"] = "1"
    else:
        os.environ.pop("KUNPENG_AFFINITY_VLLM_FORCE_GENERIC", None)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm.plugins import load_general_plugins
    from vllm.utils import numa_utils

    load_general_plugins()
    hook = numa_utils.configure_subprocess
    if not hasattr(hook, "__kunpeng_affinity_original__"):
        raise RuntimeError("kunpeng affinity vLLM hook was not installed")

    native_query_calls = 0

    def controlled_native_query():
        nonlocal native_query_calls
        native_query_calls += 1
        if verification_path == "forced-generic":
            raise AssertionError("vLLM native GPU NUMA query was called")
        return None

    numa_utils.get_auto_numa_nodes = controlled_native_query
    config = _config()
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    with hook(config, local_rank=0, process_kind="worker"):
        process = context.Process(target=_child, args=(send_connection,))
        process.start()
    send_connection.close()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        process.close()
        receive_connection.close()
        _stop_spawn_resource_tracker()
        raise RuntimeError("dummy child did not exit within 30 seconds")
    if process.exitcode != 0:
        exitcode = process.exitcode
        process.close()
        receive_connection.close()
        _stop_spawn_resource_tracker()
        raise RuntimeError(f"dummy child exited with status {exitcode}")
    result = receive_connection.recv()
    receive_connection.close()
    process.close()
    _stop_spawn_resource_tracker()

    nodes = config.parallel_config.numa_bind_nodes
    if not nodes:
        raise RuntimeError("generic NUMA nodes were not committed to vLLM config")
    expected_cpus: set[int] = set()
    for node in set(nodes):
        path = f"/sys/devices/system/node/node{node}/cpulist"
        with open(path, encoding="ascii") as cpulist_file:
            expected_cpus.update(parse_cpulist(cpulist_file.read()))
    expected_cpus.intersection_update(os.sched_getaffinity(0))
    actual_cpus = set(parse_cpulist(result["cpus_allowed_list"]))
    if actual_cpus != expected_cpus:
        raise RuntimeError(
            "child CPU affinity mismatch: "
            f"expected={format_cpulist(expected_cpus)} "
            f"actual={result['cpus_allowed_list']}"
        )

    expected_nodes = {str(node) for node in nodes}
    actual_memory_nodes = set(result["numactl"].get("membind", "").split())
    memory_policy_verified = (
        result["numactl"].get("policy") == "bind"
        and actual_memory_nodes == expected_nodes
    )
    expected_native_calls = 0 if verification_path == "forced-generic" else 1
    if native_query_calls != expected_native_calls:
        raise RuntimeError(
            f"native query call mismatch: expected={expected_native_calls} "
            f"actual={native_query_calls}"
        )

    print(
        json.dumps(
            {
                "status": "success",
                "verification_path": verification_path,
                "native_numa_query_calls": native_query_calls,
                "generic_fallback_verified": verification_path == "auto-fallback",
                "generic_nodes": nodes,
                "expected_cpus": format_cpulist(expected_cpus),
                "memory_policy_verified": memory_policy_verified,
                "child": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
