from pathlib import Path
import tempfile
import unittest

from kernels.build_metadata import (
    address_hash_metadata,
    artifact_metadata,
    sha256_file,
    source_bundle_sha256,
)


class AddressHashMetadataTest(unittest.TestCase):
    def test_defaults_are_unhashed(self):
        self.assertEqual(
            address_hash_metadata({}),
            {
                "mu_addr_hash": 0,
                "mu_addr_hash_parameters": {
                    "enabled": False,
                    "elf_base": "0x0",
                    "size": "0x80000000",
                    "unit_bytes": 32,
                    "slices": 4,
                    "gpu_host_base": "0x100000000",
                    "required_load_path": "unhashed-target-or-hardware-routed-load",
                },
            },
        )

    def test_custom_direct_load_settings(self):
        metadata = address_hash_metadata(
            {
                "MU_ADDR_HASH": "1",
                "MU_ADDR_HASH_SIZE": "0x40000000",
                "MU_ADDR_HASH_UNIT": "64",
                "MU_ADDR_HASH_SLICES": "8",
            }
        )
        self.assertEqual(metadata["mu_addr_hash"], 1)
        self.assertEqual(
            metadata["mu_addr_hash_parameters"],
            {
                "enabled": True,
                "elf_base": "0x0",
                "size": "0x40000000",
                "unit_bytes": 64,
                "slices": 8,
                "gpu_host_base": "0x100000000",
                "required_load_path": "direct-memory-load",
            },
        )

    def test_rejects_invalid_settings(self):
        for settings in (
            {"MU_ADDR_HASH": "2"},
            {"MU_ADDR_HASH_SIZE": "3"},
            {"MU_ADDR_HASH_UNIT": "0"},
            {"MU_ADDR_HASH_SLICES": "1"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                address_hash_metadata(settings)

    def test_labeled_source_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "kernels/stream"
            library = root / "lib"
            workload.mkdir(parents=True)
            library.mkdir()
            (workload / "kernel.radiance.elf").write_bytes(b"device")
            (workload / "kernel.radiance.hashed.elf").write_bytes(b"hashed")
            (workload / "kernel.soc.elf").write_bytes(b"fused")
            (library / "libmuonrt.a").write_bytes(b"runtime")
            first = source_bundle_sha256(
                root, [workload / "kernel.soc.elf", workload / "kernel.radiance.elf"]
            )
            second = source_bundle_sha256(
                root, [workload / "kernel.radiance.elf", workload / "kernel.soc.elf"]
            )
            self.assertEqual(first, second)
            records = artifact_metadata(root, workload)
            self.assertEqual(records["kernel.soc.elf"]["sha256"],
                             sha256_file(workload / "kernel.soc.elf"))
            self.assertNotIn("kernel.radiance.hashed.elf", records)
            self.assertIn(
                "kernel.radiance.hashed.elf",
                artifact_metadata(root, workload, include_hashed=True),
            )


if __name__ == "__main__":
    unittest.main()
