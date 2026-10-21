"""Validated build settings shared by generated workload manifests."""

from __future__ import annotations

import os
from collections.abc import Mapping
import hashlib
from pathlib import Path


COMPLETION_PROTOCOL = "per-core-barrier-l0d-fence-mailbox"


def _integer(settings: Mapping[str, str], name: str, default: int) -> int:
    raw = settings.get(name, str(default))
    try:
        return int(raw, 0)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def address_hash_metadata(
    settings: Mapping[str, str] | None = None,
) -> dict:
    """Return the effective ``common.mk`` GPU-address-hash settings.

    STREAM and Spatter invoke ``make`` with a copy of their environment, so
    these are the exact inputs the Makefile consumes.  Values are normalized
    to integers/hex strings to keep manifests comparable across shells.
    """

    env = os.environ if settings is None else settings
    enabled_value = _integer(env, "MU_ADDR_HASH", 0)
    if enabled_value not in (0, 1):
        raise ValueError("MU_ADDR_HASH must be 0 or 1")
    size = _integer(env, "MU_ADDR_HASH_SIZE", 0x80000000)
    unit = _integer(env, "MU_ADDR_HASH_UNIT", 32)
    slices = _integer(env, "MU_ADDR_HASH_SLICES", 4)
    _power_of_two(size, "MU_ADDR_HASH_SIZE")
    _power_of_two(unit, "MU_ADDR_HASH_UNIT")
    _power_of_two(slices, "MU_ADDR_HASH_SLICES")
    if slices < 2 or unit * slices > size:
        raise ValueError("address hash needs at least two slices and one unit per slice")

    enabled = bool(enabled_value)
    return {
        "mu_addr_hash": enabled_value,
        "mu_addr_hash_parameters": {
            "enabled": enabled,
            "elf_base": "0x0",
            "size": f"0x{size:x}",
            "unit_bytes": unit,
            "slices": slices,
            "gpu_host_base": "0x100000000",
            "required_load_path": (
                "direct-memory-load" if enabled else "unhashed-target-or-hardware-routed-load"
            ),
        },
    }


def sha256_file(path: Path) -> str:
    """Hash a build input or artifact without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(repo_root: Path, paths: list[Path]) -> str:
    """Hash labeled source files so identical bytes at new paths do not collide."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        label = path.relative_to(repo_root).as_posix().encode()
        digest.update(len(label).to_bytes(4, "little"))
        digest.update(label)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(
    repo_root: Path, workload_root: Path, *, include_hashed: bool = False,
) -> dict:
    """Record the exact linked device, fused host, and Muon runtime artifacts."""

    candidates = {
        "kernel.radiance.elf": workload_root / "kernel.radiance.elf",
        "kernel.soc.elf": workload_root / "kernel.soc.elf",
        "lib/libmuonrt.a": repo_root / "lib/libmuonrt.a",
    }
    if include_hashed:
        candidates["kernel.radiance.hashed.elf"] = (
            workload_root / "kernel.radiance.hashed.elf"
        )
    return {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in candidates.items() if path.is_file()
    }
