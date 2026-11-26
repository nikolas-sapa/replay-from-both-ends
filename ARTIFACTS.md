# Local training provenance

- Runtime: Apple Silicon, Python 3.12, MLX and MLX-LM 0.32.2.
- Base asset: `mlx-community/gemma-3-270m-it-bf16`, revision `c806ef3a4ed971bd75aaee3346e0fef808512f03`.
- Run: 160 optimizer updates, rank-8 LoRA on four layers, seed 19, no paid service. See [README](README.md#reproduce) for the command.
- The base model, generated data, adapter, and training log stay outside Git. The hashes below identify the exact local artifacts used, not a guarantee of byte-identical retraining.

| Artifact | SHA-256 |
| --- | --- |
| `model.safetensors` | `0e2acba196efd538420b12060013cab1b11344fb8ebfd6a1c5813cafc2f79162` |
| `run/adapter/adapters.safetensors` | `3c45bb4262596197032488ed6fa2280378acaf9e7bc486fd5d8a618103b7fb7a` |
| `run/data/train.jsonl` | `5be237047cb711bf7ce971100cf9ba5439f9bb1ad4ecf40beb80c11276b146d5` |
| `run/data/valid.jsonl` | `68813e2d3e749ed62cd6ef65af6621f9e64eb73a8958f0f2d3739182f81a927b` |
