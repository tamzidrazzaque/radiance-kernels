import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_interval import (
    COMPLETION_PROTOCOL,
    completion_metadata,
    parse_uart,
    parse_uart_evidence,
    require_completion_protocol,
)


class ParseUartTest(unittest.TestCase):
    @staticmethod
    def report(clusters=1):
        lines = []
        starts = []
        ends = []
        for cluster in range(clusters):
            for core in range(2):
                start = 100 + cluster * 20 + core * 5
                end = 200 + cluster * 30 + core * 7
                starts.append(start)
                ends.append(end)
                lines.append(
                    f"cluster {cluster} core {core}: start={start} end={end} "
                    f"cycles={end - start}"
                )
        lines.append(f"kernel_cycles={max(ends) - min(starts)}")
        return "\n".join(lines) + "\n"

    @classmethod
    def passing_uart(cls, clusters=1, interval=1234):
        return (cls.report(clusters) +
                f"HOST_RELEASE_TO_DONE_CYCLES={interval}\nSimulation complete.\n"
                "*** PASSED *** after 5678 cycles\nCOMMAND_EXIT_CODE=\"0\"\n")

    def test_pass_with_interval(self):
        self.assertEqual(parse_uart(self.passing_uart()), (5678, 1234))

    def test_two_cluster_mailbox_evidence(self):
        evidence = parse_uart_evidence(
            self.passing_uart(clusters=2), expected_clusters=2,
        )
        self.assertEqual(evidence.muon_clusters, 2)
        self.assertEqual(evidence.muon_cores_per_cluster, 2)
        self.assertEqual(evidence.mailbox_end_reports, 4)
        self.assertEqual(evidence.kernel_cycles, 137)
        metadata = completion_metadata(evidence)
        self.assertEqual(metadata["completion_protocol"], COMPLETION_PROTOCOL)
        self.assertEqual(
            metadata["completion_protocol_evidence"]["mailbox_end_reports"], 4,
        )

    def test_reject_missing_or_duplicate_interval(self):
        valid = self.passing_uart()
        cases = (
            valid.replace("HOST_RELEASE_TO_DONE_CYCLES=1234\n", ""),
            valid.replace(
                "HOST_RELEASE_TO_DONE_CYCLES=1234\n",
                "HOST_RELEASE_TO_DONE_CYCLES=1234\n" * 2,
            ),
            valid.replace("HOST_RELEASE_TO_DONE_CYCLES=1234", "HOST_RELEASE_TO_DONE_CYCLES=0"),
        )
        for uart in cases:
            with self.subTest(uart=uart), self.assertRaises(ValueError):
                parse_uart(uart)

    def test_reject_missing_duplicate_or_incomplete_mailbox_report(self):
        old_uart = ("HOST_RELEASE_TO_DONE_CYCLES=1234\nSimulation complete.\n"
                    "*** PASSED *** after 5678 cycles\nCOMMAND_EXIT_CODE=\"0\"\n")
        with self.assertRaisesRegex(ValueError, "drained per-core mailbox"):
            parse_uart(old_uart)

        duplicate = self.passing_uart().replace(
            "cluster 0 core 1:",
            "cluster 0 core 0: start=100 end=200 cycles=100\ncluster 0 core 1:",
        )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            parse_uart(duplicate)

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            parse_uart_evidence(self.passing_uart(), expected_clusters=2)

    def test_reject_inconsistent_mailbox_timing(self):
        bad_core = self.passing_uart().replace("cycles=100", "cycles=99", 1)
        with self.assertRaisesRegex(ValueError, "per-core mailbox cycle"):
            parse_uart(bad_core)

        bad_kernel = self.passing_uart().replace("kernel_cycles=107", "kernel_cycles=106")
        with self.assertRaisesRegex(ValueError, "kernel cycle report"):
            parse_uart(bad_kernel)

        zero_end = self.passing_uart().replace("end=200 cycles=100", "end=0 cycles=4294967196")
        with self.assertRaisesRegex(ValueError, "nonzero uint32"):
            parse_uart(zero_end)

    def test_reject_failed_guest(self):
        uart = (self.report() + "HOST_RELEASE_TO_DONE_CYCLES=1234\nSimulation complete.\n"
                "*** FAILED *** (code = 1) after 5678 cycles\nCOMMAND_EXIT_CODE=\"1\"\n")
        with self.assertRaises(ValueError):
            parse_uart(uart)

    def test_require_exact_manifest_protocol_and_topology(self):
        build = {
            "completion_protocol": COMPLETION_PROTOCOL,
            "muon_clusters": 2,
            "muon_cores_per_cluster": 2,
            "muon_total_cores": 4,
        }
        self.assertEqual(require_completion_protocol(build), (2, 2))

        for update in (
            {"completion_protocol": "all-finished"},
            {"muon_clusters": 3, "muon_total_cores": 6},
            {"muon_cores_per_cluster": 1, "muon_total_cores": 2},
            {"muon_total_cores": 3},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                require_completion_protocol(build | update)


if __name__ == "__main__":
    unittest.main()
