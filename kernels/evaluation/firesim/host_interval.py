#!/usr/bin/env python3
"""Archive a passing U250 host release-to-completion timing experiment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

from capture import HWDB_SHA256, sha256, snapshot
from splice_device import device_image


PASS = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) cycles")
FAIL = re.compile(r"\*\*\* FAILED \*\*\*")
EXIT = re.compile(r'COMMAND_EXIT_CODE="(\d+)"')
INTERVAL = re.compile(r"^HOST_RELEASE_TO_DONE_CYCLES=(\d+)[ \t\r]*$", re.MULTILINE)
CORE_REPORT = re.compile(
    r"^cluster (\d+) core (\d+): start=(\d+) end=(\d+) cycles=(\d+)[ \t\r]*$",
    re.MULTILINE,
)
KERNEL_CYCLES = re.compile(r"^kernel_cycles=(\d+)[ \t\r]*$", re.MULTILINE)
COMPLETION_PROTOCOL = "per-core-barrier-l0d-fence-mailbox"
SUPPORTED_CLUSTERS = (1, 2)
UINT32_MASK = (1 << 32) - 1


@dataclass(frozen=True)
class UartEvidence:
    whole_program_cycles: int
    host_release_to_done_cycles: int
    muon_clusters: int
    muon_cores_per_cluster: int
    mailbox_end_reports: int
    kernel_cycles: int


def require_completion_protocol(build: Mapping[str, object]) -> tuple[int, int]:
    """Validate and return the selected cluster/core topology from a build manifest."""
    if build.get("completion_protocol") != COMPLETION_PROTOCOL:
        raise ValueError(
            f"ELF build must declare completion_protocol={COMPLETION_PROTOCOL!r}"
        )
    clusters = build.get("muon_clusters")
    cores = build.get("muon_cores_per_cluster")
    total = build.get("muon_total_cores")
    if type(clusters) is not int or clusters not in SUPPORTED_CLUSTERS:
        raise ValueError(f"ELF build has unsupported muon_clusters={clusters!r}")
    if type(cores) is not int or cores != 2:
        raise ValueError(f"ELF build must select two Muon cores per cluster, got {cores!r}")
    if type(total) is not int or total != clusters * cores:
        raise ValueError("ELF build has inconsistent muon_total_cores")
    return clusters, cores


def parse_uart_evidence(
    uart: str,
    *,
    expected_clusters: int | None = None,
    expected_cores_per_cluster: int = 2,
) -> UartEvidence:
    """Validate a passing run and its drained per-core mailbox report."""
    passes = list(PASS.finditer(uart))
    intervals = list(INTERVAL.finditer(uart))
    exits = EXIT.findall(uart)
    if (len(passes) != 1 or len(intervals) != 1 or exits != ["0"] or
            FAIL.search(uart) or "Simulation complete." not in uart):
        raise ValueError("UART lacks one passing guest, one timing interval, or clean driver exit")

    interval = int(intervals[0].group(1))
    if interval <= 0:
        raise ValueError("release-to-completion interval must be positive")

    reports = list(CORE_REPORT.finditer(uart))
    if not reports:
        raise ValueError("UART lacks drained per-core mailbox completion reports")
    if (type(expected_cores_per_cluster) is not int or
            expected_cores_per_cluster <= 0):
        raise ValueError("expected cores per cluster must be positive")

    keys = [(int(match.group(1)), int(match.group(2))) for match in reports]
    if expected_clusters is None:
        cluster_ids = sorted({cluster for cluster, _ in keys})
        if not cluster_ids or cluster_ids != list(range(cluster_ids[-1] + 1)):
            raise ValueError("mailbox report cluster ids are not contiguous from zero")
        expected_clusters = len(cluster_ids)
    elif type(expected_clusters) is not int or expected_clusters <= 0:
        raise ValueError("expected clusters must be positive")

    expected_keys = [
        (cluster, core)
        for cluster in range(expected_clusters)
        for core in range(expected_cores_per_cluster)
    ]
    if keys != expected_keys:
        raise ValueError(
            f"mailbox reports do not exactly cover expected cores {expected_keys!r}"
        )

    starts: list[int] = []
    ends: list[int] = []
    for match in reports:
        start, end, cycles = map(int, match.groups()[2:])
        if start <= 0 or end <= 0 or start > UINT32_MASK or end > UINT32_MASK:
            raise ValueError("mailbox START/END stamps must be nonzero uint32 values")
        if cycles <= 0 or cycles > UINT32_MASK or cycles != ((end - start) & UINT32_MASK):
            raise ValueError("per-core mailbox cycle count is inconsistent with START/END")
        starts.append(start)
        ends.append(end)

    kernel_matches = list(KERNEL_CYCLES.finditer(uart))
    if len(kernel_matches) != 1:
        raise ValueError("UART must contain exactly one drained-mailbox kernel cycle report")
    kernel_cycles = int(kernel_matches[0].group(1))
    expected_kernel_cycles = (max(ends) - min(starts)) & UINT32_MASK
    if (kernel_cycles <= 0 or kernel_cycles > UINT32_MASK or
            kernel_cycles != expected_kernel_cycles):
        raise ValueError("kernel cycle report is inconsistent with per-core START/END stamps")
    if not (reports[-1].end() <= kernel_matches[0].start() < intervals[0].start() <
            passes[0].start()):
        raise ValueError("mailbox report, timing interval, and pass marker are out of order")

    return UartEvidence(
        whole_program_cycles=int(passes[0].group(1)),
        host_release_to_done_cycles=interval,
        muon_clusters=expected_clusters,
        muon_cores_per_cluster=expected_cores_per_cluster,
        mailbox_end_reports=len(reports),
        kernel_cycles=kernel_cycles,
    )


def parse_uart(uart: str) -> tuple[int, int]:
    """Return the historical cycle pair after validating drained-mailbox evidence."""
    evidence = parse_uart_evidence(uart)
    return evidence.whole_program_cycles, evidence.host_release_to_done_cycles


def completion_metadata(evidence: UartEvidence) -> dict[str, object]:
    """Return the explicit completion-protocol portion of an archive result."""
    return {
        "completion_protocol": COMPLETION_PROTOCOL,
        "completion_protocol_evidence": {
            "uart_report": "nightly_host_report per-core START/END mailboxes",
            "muon_clusters": evidence.muon_clusters,
            "muon_cores_per_cluster": evidence.muon_cores_per_cluster,
            "mailbox_end_reports": evidence.mailbox_end_reports,
            "kernel_cycles_gpu_mcycle": evidence.kernel_cycles,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--untimed-elf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, default=Path("/scratch/firesim_queue/jobs"))
    parser.add_argument("--plan", type=Path,
                        default=Path(__file__).resolve().parent / "host-timing/plan.csv")
    args = parser.parse_args()

    source = args.elf.resolve()
    control = args.untimed_elf.resolve()
    with args.plan.open(newline="") as stream:
        planned = {int(row["job_id"]): row for row in csv.DictReader(stream)}
    row = planned[args.job]
    if (Path(row["timed_elf"]).resolve() != source or
            Path(row["untimed_elf"]).resolve() != control or
            row["timed_elf_sha256"] != sha256(source) or
            row["untimed_elf_sha256"] != sha256(control) or
            row["host_check"] != "full-digest-and-guards" or
            row["expected_uart"] != "pass"):
        raise ValueError("ELFs or expected outcome differ from pinned plan")
    build_path = source.parent / "result.json"
    build = json.loads(build_path.read_text())
    if build.get("host_timing_requested") is not True or build.get("host_full_readback") is not True:
        raise ValueError("ELF build did not request host timing and complete readback")
    expected_clusters, expected_cores = require_completion_protocol(build)
    if build.get("expected_digest") != row["expected_digest"]:
        raise ValueError("ELF expected digest differs from pinned plan")
    if device_image(source) != device_image(control):
        raise ValueError("timed and untimed RV32 device images differ")

    job_dir = args.jobs_root / str(args.job)
    request_path = job_dir / "runworkload-full.json"
    request = json.loads(request_path.read_text())
    if (request.get("job_id") != args.job or
            Path(request.get("stage_from", "")).resolve() != source or
            request.get("hwdb_config_artifact_sha256") != HWDB_SHA256):
        raise ValueError("queue request, source ELF, or pinned HWDB differs")
    # The deploy/workloads copy is shared by sequential jobs with one workload
    # name. Verify this job's own retained copy, not the mutable deploy file.
    job_elf = (job_dir / "simulation/sim_slot_0/rsyncdir" /
               f"{request['workload']}0-{request['bootbinary']}")
    if not job_elf.is_file() or sha256(job_elf) != sha256(source):
        raise ValueError("this job's FireSim ELF is missing or differs from build")
    uart_path = job_dir / "simulation/sim_slot_0/uartlog"
    evidence = parse_uart_evidence(
        uart_path.read_text(errors="replace"),
        expected_clusters=expected_clusters,
        expected_cores_per_cluster=expected_cores,
    )
    whole_program = evidence.whole_program_cycles
    interval = evidence.host_release_to_done_cycles
    stdout = (job_dir / "stdout.log").read_text(errors="replace")
    if "Flashing FPGA Slot: 0" not in stdout:
        raise ValueError("queue stdout does not show U250 flash")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot(job_dir, out / "raw")
    shutil.copy2(source, out / "kernel.soc.elf")
    shutil.copy2(control, out / "untimed-kernel.soc.elf")
    shutil.copy2(build_path, out / "build-result.json")
    shutil.copy2(source.parent / "build.log", out / "build.log")
    result = {
        "job_id": args.job,
        "guest_check": "pass-full-output-digest-and-guards",
        **completion_metadata(evidence),
        "host_release_to_done_cycles": interval,
        "firesim_target_cycles_whole_program": whole_program,
        "cycle_domain": "RV64 host rdcycle; GPU clock relation uncalibrated",
        "interval_includes": (
            "reset-release MMIO, Muon work, final block barrier, per-core L0d fences, "
            "and END-mailbox polling"
        ),
        "interval_excludes": (
            "later mailbox UART reporting, output readback and digest, earlier boot and "
            "initialization"
        ),
        "timed_elf_sha256": sha256(source),
        "job_elf_sha256": sha256(job_elf),
        "untimed_elf_sha256": sha256(control),
        "rv32_device_image_equal": True,
        "build_manifest_sha256": sha256(build_path),
        "queue_request_sha256": sha256(request_path),
        "uart_sha256": sha256(uart_path),
        "hwdb_sha256": HWDB_SHA256,
        "raw_artifacts": {p.relative_to(out).as_posix(): sha256(p)
                          for p in sorted((out / "raw").rglob("*")) if p.is_file()},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
