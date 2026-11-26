"""RBE1 archive CLI for Connect Three endpoint-compressed move records.

Usage: python archive.py {encode,decode} INPUT OUTPUT

The uncompressed side is NDJSON, one record per line:
  {"start": [[...], [...], [...], [...]], "moves": [c0, c1, c2, c3]}
"""

import hashlib
import json
import sys

from replay import decode, encode, histories, pack_boards, play, unpack_boards

MAGIC = b"RBE1"
MAX_RECORDS = 2048
DIGEST_LEN = 32
MOVES_PER_RECORD = 4
MAX_NDJSON_LINE = 4096
# Header (12) + digest (32) + 2048 snapshot pairs (8B each) + payload bits.
# Every move row has at most 4 uniform histories, so the interval is a
# 4-adic cell and nbits <= 2*count <= 4096.
MAX_NBITS = 2 * MAX_RECORDS
MAX_ARCHIVE_BYTES = 44 + MAX_RECORDS * 8 + (MAX_NBITS + 7) // 8


def _check_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _board_from_start(start):
    if not isinstance(start, list) or len(start) != 4:
        raise ValueError("start must be a list of 4 columns")
    cols = []
    for col in start:
        if not isinstance(col, list) or len(col) > 4:
            raise ValueError("each start column must be a list of at most 4 cells")
        if any(not _check_int(v) or v not in (1, 2) for v in col):
            raise ValueError("start cells must be ints 1 or 2")
        cols.append(tuple(col))
    return tuple(cols)


def _end_from_moves(start_board, moves):
    if not isinstance(moves, list) or len(moves) != MOVES_PER_RECORD:
        raise ValueError(f"moves must be a list of exactly {MOVES_PER_RECORD} columns")
    if any(not _check_int(m) or not 0 <= m < 4 for m in moves):
        raise ValueError("moves must be zero-based columns 0..3")
    board = start_board
    for m in moves:
        board = play(board, m)
    return board


def _check_record(rec):
    if not isinstance(rec, dict) or set(rec.keys()) != {"start", "moves"}:
        raise ValueError("record must have exactly keys 'start' and 'moves'")
    start_board = _board_from_start(rec["start"])
    return start_board, _end_from_moves(start_board, rec["moves"])


def encode_archive(records) -> bytes:
    if len(records) > MAX_RECORDS:
        raise ValueError(f"too many records (max {MAX_RECORDS})")
    boards = []
    rows = []
    ranks = []
    for rec in records:
        start_board, end_board = _check_record(rec)
        hs = histories(start_board, end_board, MOVES_PER_RECORD)
        try:
            rank = hs.index(tuple(rec["moves"]))
        except ValueError:
            raise ValueError("impossible history for record") from None
        boards.append(pack_boards(start_board, end_board))
        rows.append(tuple(1 for _ in hs))
        ranks.append(rank)
    payload, nbits = encode(tuple(ranks), rows)
    head = MAGIC + len(records).to_bytes(4, "little") + nbits.to_bytes(4, "little")
    body = head + b"".join(boards) + payload
    return body + hashlib.sha256(body).digest()


def decode_archive(data: bytes) -> list[dict]:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("archive too large")
    if len(data) < 4 + 4 + 4 + DIGEST_LEN:
        raise ValueError("archive too short")
    count = int.from_bytes(data[4:8], "little")
    nbits = int.from_bytes(data[8:12], "little")
    if count > MAX_RECORDS:
        raise ValueError(f"too many records (max {MAX_RECORDS})")
    if nbits > 2 * count:
        raise ValueError("declared payload bits exceed bound")
    body, digest = data[:-DIGEST_LEN], data[-DIGEST_LEN:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError("digest mismatch")
    if body[:4] != MAGIC:
        raise ValueError("bad magic")
    boards_start = 12
    boards_end = boards_start + 8 * count
    payload_len = (nbits + 7) // 8
    if len(body) != boards_end + payload_len:
        raise ValueError("archive length does not match header")
    pairs = [
        unpack_boards(body[boards_start + 8 * i : boards_start + 8 * i + 8])
        for i in range(count)
    ]
    hs_list = [histories(s, e, MOVES_PER_RECORD) for s, e in pairs]
    if any(len(hs) == 0 for hs in hs_list):
        raise ValueError("impossible history in archive")
    rows = [tuple(1 for _ in hs) for hs in hs_list]
    payload = body[boards_end:]
    ranks = decode(payload, nbits, rows)
    out = []
    for (s, _e), hs, rank in zip(pairs, hs_list, ranks):
        out.append(
            {
                "start": [list(col) for col in s],
                "moves": list(hs[rank]),
            }
        )
    return out


def _read_ndjson(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        line_no = 0
        while True:
            line = f.readline(MAX_NDJSON_LINE + 1)
            if line == "":
                break
            line_no += 1
            if len(line) > MAX_NDJSON_LINE:
                raise ValueError(f"line {line_no}: exceeds {MAX_NDJSON_LINE} chars")
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_no}: invalid JSON") from e
            if len(records) > MAX_RECORDS:
                raise ValueError(f"too many records (max {MAX_RECORDS})")
    return records


def _write_ndjson(path, records):
    with open(path, "x", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_archive(path, data):
    with open(path, "xb") as f:
        f.write(data)


def main(argv):
    if len(argv) != 4 or argv[1] not in ("encode", "decode"):
        print("usage: archive.py {encode,decode} INPUT OUTPUT", file=sys.stderr)
        return 2
    mode, src, dst = argv[1], argv[2], argv[3]
    try:
        if mode == "encode":
            data = encode_archive(_read_ndjson(src))
            _write_archive(dst, data)
        else:
            with open(src, "rb") as f:
                raw = f.read(MAX_ARCHIVE_BYTES + 1)
            _write_ndjson(dst, decode_archive(raw))
    except FileExistsError:
        print(f"error: output exists: {dst}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
