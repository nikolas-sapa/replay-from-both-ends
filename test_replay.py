import random
import unittest

from replay import (
    EMPTY,
    decode,
    decode_online,
    encode,
    histories,
    legal,
    pack_boards,
    play,
    unpack_boards,
)


def board_from_columns(*cols):
    return tuple(tuple(c) for c in cols)


class GameTests(unittest.TestCase):
    def test_legal_start(self):
        self.assertEqual(legal(EMPTY), (0, 1, 2, 3))

    def test_play_and_alternation(self):
        b = play(EMPTY, 2)
        self.assertEqual(b, board_from_columns((), (), (1,), ()))
        b = play(b, 2)
        self.assertEqual(b[2], (1, 2))
        b = play(b, 0)
        self.assertEqual(b[0], (1,))

    def test_play_full_column_rejected(self):
        b = board_from_columns((1, 2, 1, 2), (), (), ())
        with self.assertRaises(ValueError):
            play(b, 0)

    def test_vertical_win_terminal(self):
        b = EMPTY
        for c in (0, 1, 0, 1, 0):
            b = play(b, c)
        self.assertEqual(legal(b), ())
        with self.assertRaises(ValueError):
            play(b, 2)

    def test_horizontal_win_terminal(self):
        b = EMPTY
        for c in (0, 3, 1, 3, 2):
            b = play(b, c)
        self.assertEqual(legal(b), ())

    def test_histories_known_ambiguity(self):
        # Positions 0 and 2 are same parity, so columns 0 and 2 are
        # interchangeable: two distinct orders reach the same end.
        end = board_from_columns((1,), (2,), (1,), ())
        self.assertEqual(histories(EMPTY, end, 3), ((0, 1, 2), (2, 1, 0)))

    def test_histories_four_ply_four_orders(self):
        # Four plies landing one token in every column. Same-parity slots
        # {0,2} (token 1) and {1,3} (token 2) each swap freely: 2 x 2 = 4.
        end = board_from_columns((1,), (2,), (1,), (2,))
        self.assertEqual(
            histories(EMPTY, end, 4),
            ((0, 1, 2, 3), (0, 3, 2, 1), (2, 1, 0, 3), (2, 3, 0, 1)),
        )

    def test_histories_winning_last_move_in_segment(self):
        end = EMPTY
        for c in (0, 1, 0, 1, 0):  # final move is a connect-three win for 1
            end = play(end, c)
        self.assertEqual(legal(end), ())  # target is terminal
        self.assertEqual(histories(EMPTY, end, 5), ((0, 1, 0, 1, 0),))

    def test_histories_single(self):
        end = board_from_columns((), (), (1,), ())
        self.assertEqual(histories(EMPTY, end, 1), ((2,),))

    def test_histories_empty(self):
        self.assertEqual(histories(EMPTY, EMPTY, 0), ((),))
        self.assertEqual(histories(EMPTY, EMPTY, 1), ())

    def test_histories_prunes_wrong_direction(self):
        end = board_from_columns((1,), (), (), ())
        self.assertEqual(histories(EMPTY, end, 2), ())

    def test_histories_early_terminal(self):
        b = EMPTY
        for c in (0, 1, 0, 1, 0):
            b = play(b, c)
        end = tuple(col + ((2,) if i == 3 else ()) for i, col in enumerate(b))
        self.assertEqual(histories(b, end, 1), ())

    def test_histories_token_count_mismatch(self):
        end = board_from_columns((1,), (2,), (), ())
        self.assertEqual(histories(EMPTY, end, 3), ())


class PackTests(unittest.TestCase):
    def test_pack_size(self):
        self.assertEqual(len(pack_boards(EMPTY, EMPTY)), 8)

    def test_roundtrip(self):
        b = EMPTY
        for c in (0, 2, 0, 3, 1):
            b = play(b, c)
        data = pack_boards(EMPTY, b)
        self.assertEqual(unpack_boards(data), (EMPTY, b))

    def test_known_bytes(self):
        b = board_from_columns((1, 2), (), (), ())
        data = pack_boards(b, EMPTY)
        self.assertEqual(data[:4], (0b00001001).to_bytes(4, "little"))
        self.assertEqual(data[4:], b"\x00" * 4)

    def test_rejects_wrong_size(self):
        for n in (0, 7, 9):
            with self.assertRaises(ValueError):
                unpack_boards(b"\x00" * n)

    def test_rejects_hole(self):
        word = 0b00000100  # cell0=0, cell1=1 -> hole
        data = word.to_bytes(4, "little") + b"\x00" * 4
        with self.assertRaises(ValueError):
            unpack_boards(data)

    def test_rejects_invalid_cell_value(self):
        word = 0b11  # cell0 == 3
        data = word.to_bytes(4, "little") + b"\x00" * 4
        with self.assertRaises(ValueError):
            unpack_boards(data)


class CodecTests(unittest.TestCase):
    def test_empty(self):
        payload, nbits = encode((), ())
        self.assertEqual((payload, nbits), (b"", 0))
        self.assertEqual(decode(payload, nbits, ()), ())

    def test_singleton_zero_bits(self):
        rows = [(3,), (1,)]
        payload, nbits = encode((0, 0), rows)
        self.assertEqual(nbits, 0)
        self.assertEqual(decode(payload, nbits, rows), (0, 0))

    def test_mixed_weights_with_singleton(self):
        rows = [(1, 1, 2), (4,), (7, 1)]
        symbols = (2, 0, 0)
        payload, nbits = encode(symbols, rows)
        self.assertEqual(decode(payload, nbits, rows), symbols)

    def test_rank_zero(self):
        rows = [(1, 1, 2)]
        payload, nbits = encode((0,), rows)
        self.assertEqual(decode(payload, nbits, rows), (0,))

    def test_decode_needs_no_length(self):
        rows = [(2, 1)] * 3
        payload, nbits = encode((1, 0, 1), rows)
        self.assertEqual(len(decode(payload, nbits, rows)), 3)

    def test_symbol_out_of_range_rejected(self):
        for rank in (2, -1, True):
            with self.assertRaises(ValueError):
                encode((rank,), [(1, 1)])

    def test_bad_row_rejected(self):
        for rows in ([()], [(0,)], [(1, -1)], [(-2, 3)], [(True, 1)], [{1: 2}]):
            with self.assertRaises(ValueError):
                encode((0,), rows)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            encode((0, 1), [(1, 1)])

    def test_nonzero_padding_rejected(self):
        rows = [(1, 1, 1)]
        payload, nbits = encode((1,), rows)
        self.assertEqual(nbits % 8, 1)
        bad = payload[:-1] + bytes([payload[-1] | 1])
        with self.assertRaises(ValueError):
            decode(bad, nbits, rows)

    def test_malformed_size_rejected(self):
        with self.assertRaises(ValueError):
            decode(b"", 9, [(1, 1)])
        with self.assertRaises(ValueError):
            decode(b"\x00\x00", 3, [(1, 1)])

    def test_roundtrip_random(self):
        rng = random.Random(19)
        for _ in range(400):
            nrows = rng.randrange(0, 12)
            rows = []
            symbols = []
            for _ in range(nrows):
                k = rng.randrange(1, 6)
                row = tuple(rng.randrange(1, 50) for _ in range(k))
                rows.append(row)
                symbols.append(rng.randrange(k))
            payload, nbits = encode(tuple(symbols), rows)
            self.assertLessEqual(nbits, len(payload) * 8)
            self.assertLess(len(payload) * 8 - nbits, 8)
            self.assertEqual(decode(payload, nbits, rows), tuple(symbols))

    def test_decode_online_dependent_rows(self):
        def rows_gen():
            rank1 = yield (1, 1)
            if rank1 == 0:
                yield (1, 1)
            else:
                yield (5, 9)

        payload, nbits = encode((0, 0), [(1, 1), (1, 1)])
        self.assertEqual(decode_online(payload, nbits, rows_gen()), (0, 0))

        payload, nbits = encode((1, 1), [(1, 1), (5, 9)])
        self.assertEqual(decode_online(payload, nbits, rows_gen()), (1, 1))

    def test_decode_online_empty(self):
        def rows_gen():
            return
            yield  # pragma: no cover - makes this a generator

        self.assertEqual(decode_online(b"", 0, rows_gen()), ())

    def test_heavy_imbalance_roundtrip(self):
        rows = [(1, 10_000), (999_999, 1, 1)]
        symbols = (1, 2)
        payload, nbits = encode(symbols, rows)
        self.assertEqual(decode(payload, nbits, rows), symbols)


class EndToEndTests(unittest.TestCase):
    def test_history_roundtrip_through_codec(self):
        end = board_from_columns((1,), (2,), (1,), ())
        hs = histories(EMPTY, end, 3)
        self.assertTrue(hs)
        chosen = hs[0]
        payload, nbits = encode((0,), [tuple(1 for _ in hs)])
        (idx,) = decode(payload, nbits, [tuple(1 for _ in hs)])
        self.assertEqual(hs[idx], chosen)
        packed = pack_boards(EMPTY, end)
        self.assertEqual(unpack_boards(packed), (EMPTY, end))


if __name__ == "__main__":
    unittest.main()
