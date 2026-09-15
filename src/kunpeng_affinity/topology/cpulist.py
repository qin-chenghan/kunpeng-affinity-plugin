"""Linux CPU list parsing and formatting."""

from __future__ import annotations


class CpuListError(ValueError):
    """Raised when Linux CPU list syntax is invalid."""


def parse_cpulist(value: str) -> frozenset[int]:
    """Parse Linux list syntax such as ``0-3,8,10-11``."""
    value = value.strip()
    if not value:
        return frozenset()

    cpus: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise CpuListError(f"empty CPU-list item in {value!r}")
        if "-" not in item:
            try:
                cpu = int(item)
            except ValueError as exc:
                raise CpuListError(f"invalid CPU number {item!r}") from exc
            if cpu < 0:
                raise CpuListError(f"CPU number must be non-negative: {item!r}")
            cpus.add(cpu)
            continue

        bounds = item.split("-")
        if len(bounds) != 2:
            raise CpuListError(f"invalid CPU range {item!r}")
        try:
            start, end = (int(bound) for bound in bounds)
        except ValueError as exc:
            raise CpuListError(f"invalid CPU range {item!r}") from exc
        if start < 0 or end < 0 or start > end:
            raise CpuListError(f"invalid CPU range {item!r}")
        cpus.update(range(start, end + 1))
    return frozenset(cpus)


def format_cpulist(cpus: frozenset[int] | set[int] | tuple[int, ...]) -> str:
    """Format CPUs using ascending Linux CPU-list syntax."""
    values = sorted(cpus)
    if not values:
        return ""

    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)
