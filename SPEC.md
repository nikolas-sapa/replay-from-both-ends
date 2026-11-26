# Replay From Both Ends

Given a starting and ending 4x4 Connect Three board and a known four-ply segment length, enumerate exactly the legal histories connecting them. Encode the selected history, not the moves in isolation. No model participates in move legality. A model can change the byte count, never which histories are representable.

## Acceptance checks (before implementation)

1. Reconstruct all four moves and the exact endpoint on 1,000 fixed-seed simulated test segments with all three implemented distributions: uniform forward, uniform endpoint history, and fitted first-order history. No mismatches allowed.
2. Each framed stream contains 4 bytes for segment count, 4 for payload bit length, 8 per board pair, then a whole-byte-padded payload. Report total bytes and payload bits. Hold snapshots and framing fixed. A known ambiguous board pair must offer four legal histories; a singleton takes zero payload bits in isolation.
3. Compare frozen and LoRA-tuned Gemma on the same 128 ambiguous test segments and against uniform and fitted first-order probabilities. Use the exact coder, with the same positive integer quantization for model scores. Invalid or non-finite scores fail instead of falling back. Report payload and full stream bytes, plus checkpoint staging size separately; zero replay mismatches.
4. Hold train, validation and test starting boards disjoint; reject overlapping starting boards rather than splitting repeated positions by row. Use train/validation only for training. Keep the test fixed, report bad results too. This measures simulated play, not human gameplay or chess.
5. Report cold local weight loading and probability construction separately from warm codec timing. The benchmark reuses a frozen probability table at decode for both models; it does not prove independent cross-device model reproducibility. On-device deployment and a self-contained probability sidecar are out of scope.

## Release extension

Keep the 128-case result as the initial test. Reuse the unchanged model and adapter on every ambiguous case in the fixed test split, with a first-order fit trained on the exact same segments used by LoRA. Since the extended set includes the initial 128, it is not an independent replication. No retraining or threshold selection is part of this extension.

Ship an independent model-free archive CLI. Its stream is `RBE1`, followed by two little-endian unsigned 32-bit integers for record count and payload bit length, the snapshot pairs, the most-significant-first padded rank payload, then the SHA-256 digest of every preceding byte. Up to 2,048 four-ply records are supported per archive. The decoder verifies the digest and rejects malformed input before creating the output file. This is not an authenticated format.

## Implementation order

1. Build legal play, board serialization and the exact interval coder. Round-trip every stream and test malformed padding and terminal behavior.
2. Generate disjoint samples. Use forward legal-move probabilities, endpoint-uniform, and a cheap fitted transition model. Save simulation configuration and actual serialized stream sizes.
3. Render endpoints and candidate histories for Gemma as a short multiple-choice prompt. Balance position cues by shuffling candidates per segment deterministically from the known endpoints. Fine-tune one rank-8 LoRA adapter on the train split, then compare constrained next-token distributions for frozen and tuned models. An ambiguous four-ply block has at most four histories: each player's two moves can be permuted at most two ways, and the endpoint fixes each player's final column multiset.
4. Show what saved bits in the README, include a small diagram, and identify which numbers do not include checkpoint distribution cost. Treat model-assisted coding as exploratory if a fitted non-neural baseline wins.

## Non-goals

Chess, trained gameplay agents, UI deployment, universal compression claims, paid compute, claims of research novelty, or a HALO competition submission without porting to the official track runtime.
