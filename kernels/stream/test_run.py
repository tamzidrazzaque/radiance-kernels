"""Small serial checks for the STREAM builder."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import muon_topology


class StreamBuilderTest(unittest.TestCase):
    def test_muon_topology(self):
        one = muon_topology(1)
        two = muon_topology(2)
        self.assertEqual(one["muon_total_cores"], 2)
        self.assertEqual(one["muon_hardware_warps_per_core"], 8)
        self.assertEqual(one["muon_launch_total_threads"], 128)
        self.assertEqual(two["muon_total_cores"], 4)
        self.assertEqual(two["muon_launch_total_threads"], 256)
        with self.assertRaisesRegex(ValueError, "muon_clusters"):
            muon_topology(3)


if __name__ == "__main__":
    unittest.main()
