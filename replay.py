"""4x4 gravity Connect Three plus reversible board/history codecs.

Board = tuple[tuple[int, ...], ...]: 4 columns, each a bottom-up tuple of
tokens (1 or 2). Token 1 moves first; turns alternate by total parity.
"""

from fractions import Fraction

EMPTY: tuple[tuple[int, ...], ...] = ((), (), (), ())

WIDTH = 4
HEIGHT = 4
CONNECT = 3

_DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _token(board) -> int:
    return 1 if sum(len(col) for col in board) % 2 == 0 else 2


def _winner(board) -> bool:
    for c in range(WIDTH):
        for r in range(len(board[c])):
            t = board[c][r]
            for dc, dr in _DIRS:
                if all(
                    0 <= c + k * dc < WIDTH
                    and 0 <= r + k * dr < len(board[c + k * dc])
                    and board[c + k * dc][r + k * dr] == t
                    for k in range(CONNECT)
                ):
                    return True
    return False


def _terminal(board) -> bool:
    return all(len(col) == HEIGHT for col in board) or _winner(board)


def _check_board(board) -> None:
    if len(board) != WIDTH or any(
        len(col) > HEIGHT or any(t not in (1, 2) for t in col) for col in board
    ):
        raise ValueError("invalid board")
    ones = sum(col.count(1) for col in board)
    twos = sum(col.count(2) for col in board)
    if not (0 <= ones - twos <= 1):
        raise ValueError("implausible token counts")


def legal(board) -> tuple[int, ...]:
    _check_board(board)
    if _terminal(board):
        return ()
    return tuple(c for c in range(WIDTH) if len(board[c]) < HEIGHT)


def play(board, column: int):
    _check_board(board)
    if column not in legal(board):
        raise ValueError(f"illegal column {column}")
    cols = [list(col) for col in board]
    cols[column].append(_token(board))
    return tuple(tuple(col) for col in cols)


def _is_prefix_board(board, end) -> bool:
    return all(
        len(board[c]) <= len(end[c]) and board[c] == end[c][: len(board[c])]
        for c in range(WIDTH)
    )


def histories(start, end, length: int) -> tuple[tuple[int, ...], ...]:
    _check_board(start)
    _check_board(end)
    if length < 0:
        raise ValueError("negative length")
    # Every move appends exactly one token, so a path of `length` moves exists
    # only when the token totals differ by exactly that many.
    if sum(len(c) for c in end) - sum(len(c) for c in start) != length:
        return ()
    out = []

    def dfs(board, path):
        if not _is_prefix_board(board, end):
            return
        # Accept the target even on a terminal board: a winning final move
        # inside a segment is still a legal history.
        if len(path) == length:
            if board == end:
                out.append(tuple(path))
            return
        if _terminal(board):
            return
        for c in range(WIDTH):
            if len(board[c]) < HEIGHT:
                dfs(play(board, c), path + [c])

    dfs(start, [])
    out.sort()
    return tuple(out)


def pack_boards(start, end) -> bytes:
    _check_board(start)
    _check_board(end)
    words = []
    for board in (start, end):
        w = 0
        for c in range(WIDTH):
            for r in range(HEIGHT):
                cell = board[c][r] if r < len(board[c]) else 0
                w |= cell << (2 * (c * HEIGHT + r))
        words.append(w)
    return words[0].to_bytes(4, "little") + words[1].to_bytes(4, "little")


def _unpack_one(word: int):
    cols = []
    for c in range(WIDTH):
        cells = [(word >> (2 * (c * HEIGHT + r))) & 0b11 for r in range(HEIGHT)]
        if 3 in cells:
            raise ValueError("invalid cell value")
        try:
            first_zero = cells.index(0)
        except ValueError:
            first_zero = HEIGHT
        if any(v != 0 for v in cells[first_zero:]):
            raise ValueError("hole in column")
        cols.append(tuple(cells[:first_zero]))
    board = tuple(cols)
    _check_board(board)
    return board


def unpack_boards(data: bytes):
    if len(data) != 8:
        raise ValueError("payload must be exactly 8 bytes")
    return (
        _unpack_one(int.from_bytes(data[:4], "little")),
        _unpack_one(int.from_bytes(data[4:], "little")),
    )


def _check_row(row) -> None:
    if not isinstance(row, (tuple, list)) or not row or any(
        not isinstance(w, int) or isinstance(w, bool) or w <= 0 for w in row
    ):
        raise ValueError("rows must be nonempty tuples/lists of positive int weights")


def _check_weights(weights) -> None:
    for row in weights:
        _check_row(row)


def _check_rank(rank, row) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < len(row):
        raise ValueError(f"invalid rank {rank!r}")


def _feasible_point(lo: Fraction, hi: Fraction, n: int) -> int | None:
    scale = 1 << n
    p = -(-lo.numerator * scale // lo.denominator)
    if p * hi.denominator < hi.numerator * scale:
        return p
    return None


def _shortest_point(lo: Fraction, hi: Fraction) -> tuple[int, int]:
    p = _feasible_point(lo, hi, 0)
    if p is not None:
        return p, 0
    hi_n = 1
    while _feasible_point(lo, hi, hi_n) is None:
        hi_n <<= 1
    lo_n = hi_n >> 1
    while lo_n + 1 < hi_n:
        mid = (lo_n + hi_n) >> 1
        if _feasible_point(lo, hi, mid) is None:
            lo_n = mid
        else:
            hi_n = mid
    return _feasible_point(lo, hi, hi_n), hi_n


def encode(symbols, weights) -> tuple[bytes, int]:
    if len(symbols) != len(weights):
        raise ValueError("symbols and weights length mismatch")
    _check_weights(weights)
    lo, hi = Fraction(0), Fraction(1)
    for s, row in zip(symbols, weights):
        _check_rank(s, row)
        if len(row) == 1:
            continue
        total = sum(row)
        span = hi - lo
        cum = 0
        for i, w in enumerate(row):
            if i == s:
                lo, hi = lo + Fraction(cum, total) * span, lo + Fraction(
                    cum + w, total
                ) * span
                break
            cum += w
    p, nbits = _shortest_point(lo, hi)
    nbytes = (nbits + 7) // 8
    pad = nbytes * 8 - nbits
    return (p << pad).to_bytes(nbytes, "big") if nbytes else b"", nbits


def _decode_point(payload: bytes, nbits: int) -> Fraction:
    if nbits < 0 or len(payload) != (nbits + 7) // 8:
        raise ValueError("malformed payload size")
    raw = int.from_bytes(payload, "big") if payload else 0
    pad = len(payload) * 8 - nbits
    if raw & ((1 << pad) - 1):
        raise ValueError("nonzero padding bits")
    return Fraction(raw >> pad, 1 << nbits)


def _decode_row(point: Fraction, row) -> tuple[int, Fraction]:
    _check_row(row)
    if len(row) == 1:
        return 0, point
    total = sum(row)
    scaled = point * total
    cum = 0
    for i, w in enumerate(row):
        if cum <= scaled < cum + w:
            return i, Fraction(scaled - cum, w)
        cum += w
    raise ValueError("point outside all intervals")


def decode(payload: bytes, nbits: int, weights) -> tuple[int, ...]:
    _check_weights(weights)
    point = _decode_point(payload, nbits)
    out = []
    for row in weights:
        rank, point = _decode_row(point, row)
        out.append(rank)
    return tuple(out)


def decode_online(payload: bytes, nbits: int, rows) -> tuple[int, ...]:
    point = _decode_point(payload, nbits)
    out = []
    try:
        row = next(rows)
        while True:
            rank, point = _decode_row(point, row)
            out.append(rank)
            row = rows.send(rank)
    except StopIteration:
        pass
    return tuple(out)
