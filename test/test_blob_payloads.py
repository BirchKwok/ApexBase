"""
Python API coverage for descriptor-backed BLOB payload columns.
"""

import os
import sys
import tempfile

import pytest

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


def _payload(size: int, seed: int) -> bytes:
    pattern = f"apexbase-blob-test-{seed:02d}:".encode("ascii")
    return (pattern * ((size // len(pattern)) + 1))[:size]


@pytest.mark.parametrize("blob_type", ["blob", "large_binary"])
def test_blob_type_reads_full_payloads_ranges_and_metadata(blob_type):
    """Both blob schema names should support lazy point/range reads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        client = ApexClient(dirpath=temp_dir)
        client.create_table(
            "files",
            {"name": "string", "payload": blob_type},
        )

        payloads = [
            b"small-inline-payload",
            _payload(128 * 1024, 1),
            _payload(5 * 1024 * 1024 + 123, 2),
        ]
        client.store(
            {
                "name": ["small.bin", "packed.bin", "dedicated.bin"],
                "payload": payloads,
            }
        )

        assert client.count_rows() == 3
        assert client.read_blob("payload", 1) == payloads[0]
        assert client.read_blob("payload", 2) == payloads[1]
        assert client.read_blob("payload", 3) == payloads[2]
        assert client.read_blobs("payload", [1, -1, 99, 3]) == [
            payloads[0],
            None,
            None,
            payloads[2],
        ]

        assert client.read_blob_range("payload", 1, 6, 6) == payloads[0][6:12]
        assert client.read_blob_range("payload", 2, 65530, 32) == payloads[1][
            65530:65562
        ]
        assert client.read_blob_range("payload", 3, 5 * 1024 * 1024, 64) == payloads[2][
            5 * 1024 * 1024:5 * 1024 * 1024 + 64
        ]
        assert client.read_blob_range(
            "payload", 3, len(payloads[2]) - 10, None
        ) == payloads[2][-10:]
        assert client.read_blob_ranges(
            "payload", [1, 2, -1, 3], [0, 17, 0, 4096], 11
        ) == [
            payloads[0][:11],
            payloads[1][17:28],
            None,
            payloads[2][4096:4107],
        ]

        infos = client.read_blob_infos("payload", [1, 2, 3, -1, 99])
        assert [info["mode"] if info else None for info in infos] == [
            "inline",
            "packed",
            "dedicated",
            None,
            None,
        ]
        assert [info["length"] if info else None for info in infos[:3]] == [
            len(payload) for payload in payloads
        ]
        assert client.read_blob_info("payload", 2)["locator_length"] > 0
        assert client.read_blob_descriptor("payload", 1).startswith(b"ABLB")

        rows = client.execute(
            "SELECT name, payload FROM files ORDER BY _id",
            show_internal_id=False,
        ).to_dict()
        assert [row["name"] for row in rows] == [
            "small.bin",
            "packed.bin",
            "dedicated.bin",
        ]
        assert [row["payload"] for row in rows] == payloads

        client.close()


@pytest.mark.parametrize("blob_type", ["blob", "large_binary"])
def test_blob_type_helpers_distinguish_null_and_empty_payloads_after_reopen(blob_type):
    """Blob helpers should preserve NULL separately from an empty payload."""
    with tempfile.TemporaryDirectory() as temp_dir:
        payload = _payload(96 * 1024, 3)

        client = ApexClient(dirpath=temp_dir)
        client.create_table("files", {"name": "string", "payload": blob_type})
        client.store(
            {
                "name": ["missing.bin", "empty.bin", "present.bin"],
                "payload": [None, b"", payload],
            }
        )
        client.close()

        reopened = ApexClient(dirpath=temp_dir)
        reopened.use_table("files")

        assert reopened.read_blob("payload", 1) is None
        assert reopened.read_blob("payload", 2) == b""
        assert reopened.read_blob("payload", 3) == payload
        assert reopened.read_blobs("payload", [1, 2, 3]) == [None, b"", payload]
        assert reopened.read_blob_info("payload", 1) is None
        assert reopened.read_blob_info("payload", 2)["length"] == 0
        assert reopened.read_blob_info("payload", 3)["mode"] == "packed"
        assert reopened.read_blob_ranges("payload", [1, 2, 3], [0, 0, 100], 16) == [
            None,
            b"",
            payload[100:116],
        ]

        reopened.close()
