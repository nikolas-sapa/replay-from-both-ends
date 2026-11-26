import argparse
import hashlib
import json
import math
import random
import struct
from collections import Counter
from pathlib import Path
from time import perf_counter

from replay import EMPTY, decode, encode, histories, legal, pack_boards, play, unpack_boards


PLY = 4
COUNTS = {"train": 640, "valid": 128, "test": 1000}
SEEDS = {"train": 78131, "valid": 25817, "test": 49537}
OPTIONS = "ABCD"
TOTAL = 4096


def follow(board, moves):
    for move in moves:
        board = play(board, move)
    return board


def samples():
    seen = set()
    result = {}
    for name, count in COUNTS.items():
        rng = random.Random(SEEDS[name])
        rows = []
        for _ in range(120_000):
            board = EMPTY
            prefix = rng.randint(3, 9)
            for _ in range(prefix):
                choices = legal(board)
                if not choices:
                    break
                board = play(board, rng.choice(choices))
            if board in seen:
                continue
            end = board
            moves = []
            for step in range(PLY):
                choices = legal(end)
                if not choices:
                    break
                # The same visible move pattern applies to both splits.
                if step == 0:
                    weights = [5 if c in (1, 2) else 2 for c in choices]
                else:
                    weights = [15 if c == (moves[-1] + 1) % 4 else 2 for c in choices]
                moves.append(rng.choices(choices, weights=weights)[0])
                end = play(end, moves[-1])
            if len(moves) != PLY:
                continue
            paths = histories(board, end, PLY)
            assert 1 <= len(paths) <= 4 and tuple(moves) in paths
            seen.add(board)
            rows.append((board, end, tuple(moves), paths))
            if len(rows) == count:
                break
        if len(rows) != count:
            raise RuntimeError(f"Could not make {name} split: {len(rows)}/{count}")
        result[name] = rows
    assert len(seen) == sum(COUNTS.values())
    return result


def ordered(start, end, paths):
    seed = hashlib.sha256(b"replay-options-v1" + pack_boards(start, end)).digest()
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def prompt(start, end, paths):
    options = ordered(start, end, paths)
    render = lambda board: "/".join("".join(map(str, col)) or "-" for col in board)
    choices = "\n".join(f"{option}: {' '.join(map(str, path))}"
                        for option, path in zip(OPTIONS, options))
    text = (
        "Predict which legal history the simulated player actually chose. "
        "Columns are 0 to 3; boards show bottom-up columns separated by /. "
        "Reply with exactly one option letter.\n"
        f"Before: {render(start)}\nAfter: {render(end)}\n{choices}"
    )
    return text, options


def prepare(data, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid"):
        with (dest / f"{name}.jsonl").open("x") as file:
            for start, end, moves, paths in data[name]:
                if len(paths) > 1:
                    text, options = prompt(start, end, paths)
                    record = {"prompt": text, "completion": OPTIONS[options.index(moves)]}
                    file.write(json.dumps(record) + "\n")


def transitions(rows):
    counts = Counter()
    for _, _, moves, _ in rows:
        counts.update(zip((-1, *moves[:-1]), moves))
    return counts


def fitted_weights(paths, start, counts):
    values = []
    for path in paths:
        previous, board, probability = -1, start, 1.0
        for move in path:
            choices = legal(board)
            probability *= (counts[previous, move] + 1) / sum(
                counts[previous, c] + 1 for c in choices
            )
            previous, board = move, play(board, move)
        values.append(probability)
    return quantize(values)


def quantize(values):
    if not all(math.isfinite(value) and value >= 0 for value in values) or sum(values) <= 0:
        raise ValueError("Invalid model distribution")
    return tuple(1 + round(TOTAL * val / sum(values)) for val in values)


def model_weights(rows, model_path, adapter):
    import mlx.core as mx
    from mlx_lm import load

    before_load = perf_counter()
    model, tokenizer = load(str(model_path), adapter_path=str(adapter) if adapter else None)
    load_s = perf_counter() - before_load
    before_probs = perf_counter()
    result = {}
    for start, end, _, paths in rows:
        text, options = prompt(start, end, paths)
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(chat, add_special_tokens=False)
        labels = [tokenizer.encode(chat + label, add_special_tokens=False)
                  for label in OPTIONS[:len(options)]]
        if any(len(label) != len(ids) + 1 or label[:-1] != ids for label in labels):
            raise ValueError("Option is not exactly one continuable token")
        logits = model(mx.array([ids]))[0, -1][mx.array([label[-1] for label in labels])]
        vals = mx.softmax(logits.astype(mx.float32)).tolist()
        weights = quantize(vals)
        result[pack_boards(start, end)] = tuple(weights[options.index(path)] for path in paths)
    return result, {"load_seconds": load_s, "probability_seconds": perf_counter() - before_probs}


def frame(rows, payload, bits):
    return struct.pack("<II", len(rows), bits) + b"".join(
        pack_boards(start, end) for start, end, _, _ in rows
    ) + payload


def unframe(stream):
    count, bits = struct.unpack_from("<II", stream)
    boundary = 8 + 8 * count
    if len(stream) != boundary + (bits + 7) // 8:
        raise ValueError("Invalid frame length")
    return [unpack_boards(stream[i:i + 8]) for i in range(8, boundary, 8)], stream[boundary:], bits


def endpoint_roundtrip(rows, weights_for):
    tick = perf_counter()
    symbols, weights = zip(*[
        (paths.index(moves), weights_for(start, end, paths)) for start, end, moves, paths in rows
    ])
    payload, bits = encode(symbols, weights)
    stream = frame(rows, payload, bits)
    encode_s = perf_counter() - tick
    tick = perf_counter()
    boards, payload, bits = unframe(stream)
    candidate_rows = [histories(start, end, PLY) for start, end in boards]
    ranks = decode(payload, bits, [weights_for(start, end, paths)
                                   for (start, end), paths in zip(boards, candidate_rows)])
    decoded = [paths[rank] for rank, paths in zip(ranks, candidate_rows)]
    check_replay(rows, boards, decoded)
    return {"segments": len(rows), "stream_bytes": len(stream), "payload_bits": bits,
            "encode_seconds": encode_s, "decode_seconds": perf_counter() - tick}


def forward_roundtrip(rows):
    tick = perf_counter()
    symbols, weights = [], []
    for start, _, moves, _ in rows:
        board = start
        for move in moves:
            choices = legal(board)
            symbols.append(choices.index(move))
            weights.append((1,) * len(choices))
            board = play(board, move)
    payload, bits = encode(symbols, weights)
    stream = frame(rows, payload, bits)
    encode_s = perf_counter() - tick
    tick = perf_counter()
    boards, payload, bits = unframe(stream)
    # Online decoding cannot know the choice counts before recovering earlier moves.
    decoded = []

    def forward_weights():
        for start, _ in boards:
            board, moves = start, []
            for _ in range(PLY):
                choices = legal(board)
                rank = yield (1,) * len(choices)
                move = choices[rank]
                moves.append(move)
                board = play(board, move)
            decoded.append(tuple(moves))

    from replay import decode_online

    decode_online(payload, bits, forward_weights())
    check_replay(rows, boards, decoded)
    return {"segments": len(rows), "stream_bytes": len(stream), "payload_bits": bits,
            "encode_seconds": encode_s, "decode_seconds": perf_counter() - tick}


def check_replay(rows, boards, decoded):
    if len(rows) != len(boards) or len(rows) != len(decoded):
        raise ValueError("Wrong replay count")
    for (start, end, truth, _), (actual_start, actual_end), moves in zip(rows, boards, decoded):
        if (start, end, truth) != (actual_start, actual_end, moves) or follow(start, moves) != end:
            raise ValueError("Incorrect replay")


def measure(data, model, adapter, all_ambiguous=False):
    counts = transitions(data["train"])
    rows = data["test"]
    uniform = lambda _start, _end, paths: (1,) * len(paths)
    fitted = lambda start, _end, paths: fitted_weights(paths, start, counts)
    ambiguous = [r for r in rows if len(r[3]) > 1]
    comparison = ambiguous if all_ambiguous else ambiguous[:128]
    if len(comparison) < 128:
        raise ValueError("Too few ambiguous examples")
    result = {
        "split": COUNTS, "seeds": SEEDS, "length": PLY,
        "total_ambiguous": len(ambiguous),
        "full_test": {"forward": forward_roundtrip(rows),
                      "endpoint_uniform": endpoint_roundtrip(rows, uniform),
                      "endpoint_fitted": endpoint_roundtrip(rows, fitted)},
        "ambiguous_comparison": {
            "endpoint_uniform": endpoint_roundtrip(comparison, uniform),
            "endpoint_fitted": endpoint_roundtrip(comparison, fitted),
        },
    }
    if all_ambiguous:
        training = [r for r in data["train"] if len(r[3]) > 1]
        matched_counts = transitions(training)
        result["matched_training_segments"] = len(training)
        result["ambiguous_comparison"]["endpoint_fitted_matched"] = endpoint_roundtrip(
            comparison,
            lambda start, _end, paths: fitted_weights(paths, start, matched_counts),
        )
    if model:
        if not adapter:
            raise ValueError("Supply both model and adapter for matched comparison")
        for method, checkpoint in (("gemma_frozen", None), ("gemma_lora", adapter)):
            table, timing = model_weights(comparison, model, checkpoint)
            provider = lambda start, end, _paths: table[pack_boards(start, end)]
            result["ambiguous_comparison"][method] = endpoint_roundtrip(
                comparison, provider
            ) | timing
        result["adapter_bytes"] = (adapter / "adapters.safetensors").stat().st_size
        result["base_weight_bytes"] = sum(p.stat().st_size for p in model.glob("*.safetensors"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-ambiguous", action="store_true")
    args = parser.parse_args()
    data = samples()
    if args.prepare:
        prepare(data, args.prepare)
    else:
        report = json.dumps(measure(data, args.model, args.adapter, args.all_ambiguous),
                            indent=2) + "\n"
        if args.output:
            with args.output.open("x") as file:
                file.write(report)
        print(report, end="")
