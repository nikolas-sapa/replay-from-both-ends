# Replay From Both Ends

How much of a game replay can you discard if you already know what the board looks like at both ends?

![Four legal sequences connect the same pair of Connect Three boards. A probability distribution changes the cost of encoding each sequence, not which sequence decodes.](replay.svg)

In four-ply, 4x4 gravity Connect Three, the endpoint limits which columns each player used. The codec enumerates legal histories and sends the rank of the one that happened. A unique history needs no additional payload. When several histories fit, an exact interval code can spend fewer bits on likely ones. A wrong top guess costs bits; it does not discard the true history.

## Results

All methods reconstructed every tested move exactly. Of 1,000 simulated test segments, 720 had multiple legal histories and 280 had only one.

| Method, same 1,000 segments | Move payload | Complete stream |
| --- | ---: | ---: |
| Forward legal moves, uniform | 6,758 bits | 8,853 B |
| Endpoint histories, uniform | 1,055 bits | 8,140 B |
| Endpoint histories, fitted transitions | **411 bits** | **8,060 B** |

The endpoint alone saved **713 B (8.1%)** of complete stream size. Both codecs include the same 8,008 B of snapshots and framing.

| Method, all 720 ambiguous test segments | Move payload | Complete stream |
| --- | ---: | ---: |
| Endpoint histories, uniform | 1,055 bits | 5,900 B |
| Endpoint histories, fitted transitions (all 640 train segments) | **411 bits** | **5,820 B** |
| Endpoint histories, fitted transitions (same 484 train segments as LoRA) | 413 bits | **5,820 B** |
| Frozen Gemma 3 | 1,384 bits | 5,941 B |
| LoRA-tuned Gemma 3 | 1,078 bits | 5,903 B |

Fine-tuning saved **38 B versus frozen Gemma**, but used **3 B more than uniform** and **83 B more than the fitted predictor trained on the same cases**. This extended run uses the original adapter, without any more training or test-set tuning. The local run used 484 ambiguous training examples, 95 validation examples, and 160 optimizer updates. The adapter weighs 1,693,431 B; the base weight file weighs 871,767,174 B. See the [full measurements](RESULTS_EXTENDED.json), [original 128-case pilot](RESULTS.json), and [artifact hashes](ARTIFACTS.md). Timings are from one machine.

## Encode a file

The standalone CLI uses the uniform endpoint codec and needs no model or third-party packages. A record is one four-move segment, with a starting board in four bottom-up columns and moves numbered from 0 to 3. See [the sample](segments.ndjson).

```sh
python archive.py encode segments.ndjson sample.rbe
python archive.py decode sample.rbe recovered.ndjson
```

The archive restores the board and moves, not the input's whitespace. Its `RBE1` header and SHA-256 corruption check add 36 bytes beyond the benchmark framing; the table does not include those bytes. The checksum detects accidental damage, not tampering by someone who can recompute it. Output files are never overwritten.

## Reproduce

Python 3.11 or newer is sufficient for the codec and simulated benchmark. The training comparison uses MLX-LM 0.32.2 on Apple Silicon, with the BF16 [Gemma 3 270M instruct conversion](https://huggingface.co/mlx-community/gemma-3-270m-it-bf16/tree/c806ef3a4ed971bd75aaee3346e0fef808512f03) available locally.

```sh
python -m unittest -v
python experiment.py
python experiment.py --prepare run/data
mlx_lm.lora --model "$MODEL_DIR" --train --data run/data \
  --iters 160 --batch-size 4 --num-layers 4 --max-seq-length 256 \
  --mask-prompt --learning-rate 0.0001 --val-batches 4 \
  --steps-per-report 20 --steps-per-eval 80 --save-every 160 \
  --seed 19 --adapter-path run/adapter
python experiment.py --all-ambiguous --model "$MODEL_DIR" --adapter run/adapter
```

## Scope

- All streams include the same start and end snapshots. This measures compression **with snapshots**, not against plain game logs that never store them.
- Simulated play has an explicit transition bias. A first-order model fitted on training moves is therefore a serious baseline; Gemma is not assumed to beat it.
- Both model encoders and decoders must agree on the exact quantized probabilities and canonical path order. The benchmark caches those probabilities, does not include their delivery cost in stream bytes, and reports checkpoint sizes separately.
- This is a local experiment, not a HALO submission or a claim of novelty. The original [ASC Chess](https://github.com/Anemosalo/ASC-Chess-Implementation) motivated the search for a small, testable coding trick; this implementation uses endpoint constraints rather than stashed aliases.
