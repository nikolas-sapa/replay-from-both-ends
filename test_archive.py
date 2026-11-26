import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from archive import MAGIC, MAX_ARCHIVE_BYTES, decode_archive, encode_archive

EMPTY_START = [[], [], [], []]


def rec(moves, start=None):
    return {"start": start if start is not None else EMPTY_START, "moves": moves}


class ArchiveTests(unittest.TestCase):
    def test_empty_roundtrip(self):
        data = encode_archive([])
        self.assertTrue(data.startswith(MAGIC))
        self.assertEqual(decode_archive(data), [])

    def test_singleton(self):
        # Four drops in one column: only column 0 was used, so the history
        # is unique and costs zero bits.
        records = [rec([0, 0, 0, 0])]
        self.assertEqual(decode_archive(encode_archive(records)), records)

    def test_four_histories(self):
        # end [[1],[2],[1],[2]] admits exactly 4 orders; roundtrip any of them
        movesets = [[0, 1, 2, 3], [0, 3, 2, 1], [2, 1, 0, 3], [2, 3, 0, 1]]
        records = [rec(m) for m in movesets]
        self.assertEqual(decode_archive(encode_archive(records)), records)

    def test_win_on_final_move(self):
        # start has 4 tokens -> next is token 1. Sequence [0,1,3,2] plays
        # token 2 into col 2 last, completing a vertical (2,2,2) win on the
        # segment's final move.
        start = [[1], [1], [2, 2], []]
        records = [rec([0, 1, 3, 2], start=start)]
        self.assertEqual(decode_archive(encode_archive(records)), records)

    def test_nonzero_start(self):
        start = [[1, 2, 1], [2], [], []]
        records = [rec([1, 2, 3, 2], start=start)]
        self.assertEqual(decode_archive(encode_archive(records)), records)

    def test_truncated_archive(self):
        data = encode_archive([rec([0, 1, 2, 3])])
        for n in (0, 5, len(data) - 33, len(data) - 1):
            with self.assertRaises(ValueError):
                decode_archive(data[:n])

    def test_corrupted_digest(self):
        data = bytearray(encode_archive([rec([0, 1, 2, 3])]))
        data[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            decode_archive(bytes(data))
        data = bytearray(encode_archive([rec([0, 1, 2, 3])]))
        data[20] ^= 0xFF  # corrupt body -> digest mismatch
        with self.assertRaises(ValueError):
            decode_archive(bytes(data))

    def test_bad_magic(self):
        data = bytearray(encode_archive([]))
        data[0:4] = b"XXXX"
        # recompute digest so only magic is wrong
        data[-32:] = hashlib.sha256(bytes(data[:-32])).digest()
        with self.assertRaises(ValueError):
            decode_archive(bytes(data))

    def test_invalid_records(self):
        bad = [
            {"start": EMPTY_START, "moves": [0, 1, 2, 3], "extra": 1},
            {"start": EMPTY_START},
            {"start": EMPTY_START, "moves": [0, 1, 2]},
            {"start": EMPTY_START, "moves": [0, 1, 2, 4]},
            {"start": EMPTY_START, "moves": [0, 1, 2, True]},
            {"start": [[True], [], [], []], "moves": [0, 1, 2, 3]},
            {"start": [[1, 2, 1, 2, 1], [], [], []], "moves": [0, 1, 2, 3]},
            {"start": [[3], [], [], []], "moves": [0, 1, 2, 3]},
            {"start": [[2], [2], [], []], "moves": [0, 1, 2, 3]},
            {"start": [[1, 2, 1, 2], [], [], []], "moves": [0, 1, 2, 3]},
            "not a dict",
        ]
        for r in bad:
            with self.assertRaises(ValueError, msg=f"record {r!r}"):
                encode_archive([r])

    def test_oversize(self):
        records = [rec([0, 1, 2, 3])] * 2049
        with self.assertRaises(ValueError):
            encode_archive(records)

    def test_oversize_count_field(self):
        body = MAGIC + (2049).to_bytes(4, "little") + (0).to_bytes(4, "little")
        data = body + hashlib.sha256(body).digest()
        with self.assertRaises(ValueError):
            decode_archive(data)

    def test_oversize_nbits_field(self):
        # count=1 allows at most 2 payload bits; declare more, rehash so the
        # digest is valid, and confirm the bound still rejects it.
        good = encode_archive([rec([0, 0, 0, 0])])
        body = bytearray(good[:-32])
        body[8:12] = (10).to_bytes(4, "little")
        data = bytes(body) + hashlib.sha256(bytes(body)).digest()
        with self.assertRaises(ValueError):
            decode_archive(data)

    def test_oversize_total(self):
        good = encode_archive([rec([0, 0, 0, 0])])
        with self.assertRaises(ValueError):
            decode_archive(good + b"\x00" * (MAX_ARCHIVE_BYTES - len(good) + 1))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive.py")

    def _write(self, name, text, mode="w"):
        path = os.path.join(self.dir, name)
        with open(path, mode) as f:
            f.write(text)
        return path

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, self.script, *args],
            capture_output=True,
            text=True,
        )

    def test_cli_roundtrip(self):
        src = self._write(
            "in.ndjson",
            json.dumps(rec([0, 1, 2, 3])) + "\n" + json.dumps(rec([2, 3, 0, 1])) + "\n",
        )
        arch = os.path.join(self.dir, "out.rbe")
        r = self._run("encode", src, arch)
        self.assertEqual(r.returncode, 0, r.stderr)
        back = os.path.join(self.dir, "back.ndjson")
        r = self._run("decode", arch, back)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(back) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(
            lines, [rec([0, 1, 2, 3]), rec([2, 3, 0, 1])]
        )

    def test_cli_refuses_overwrite(self):
        src = self._write("in.ndjson", json.dumps(rec([0, 1, 2, 3])) + "\n")
        arch = os.path.join(self.dir, "out.rbe")
        self.assertEqual(self._run("encode", src, arch).returncode, 0)
        r = self._run("encode", src, arch)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exists", r.stderr)

    def test_cli_no_output_on_invalid(self):
        src = self._write("bad.ndjson", '{"start": [[true],[],[],[]], "moves":[0,1,2,3]}\n')
        out = os.path.join(self.dir, "never.rbe")
        r = self._run("encode", src, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(out))

    def test_cli_oversized_ndjson_line(self):
        src = self._write("huge.ndjson", " " * 4100 + "\n")
        out = os.path.join(self.dir, "never.rbe")
        r = self._run("encode", src, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(out))

    def test_cli_decode_corrupt(self):
        arch = self._write(
            "bad.rbe",
            encode_archive([rec([0, 1, 2, 3])])[:-1],
            mode="wb",
        )
        out = os.path.join(self.dir, "never.ndjson")
        r = self._run("decode", arch, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
