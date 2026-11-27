# Halo SFT smoke test

This ports the replay task's data to Halo's chat format. The original MLX/Gemma 3 result is not a Halo result. The steps below are for a training environment with approved GPU access; no Halo training has run yet.

- Upstream reference: [Halo `9f32a5f`](https://github.com/whitecircle/halo/tree/9f32a5f707f0580d5fa0ced4182c52706e817c9f), [dataset format](https://github.com/whitecircle/halo/blob/9f32a5f707f0580d5fa0ced4182c52706e817c9f/agent-docs/data/dataset-formats.md), [Gemma 4 EP8 example](https://github.com/whitecircle/halo/blob/9f32a5f707f0580d5fa0ced4182c52706e817c9f/examples/sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml). Confirm the competition-approved version and model before using this recipe there.
- The Arrow `DatasetDict` has 484 `train` and 95 `test` rows. Halo's `test` is the original replay validation set for trainer evaluation; none of the 1,000 final benchmark segments enters this export. Do not add `test_size`, which would repartition an already split dataset.
- The smoke config preserves the reference model and EP8 settings, sets a short context, requests two optimizer updates and a checkpoint, and disables external reporting. It uses the official example's full-parameter SFT, not LoRA. It is not meant for a Mac or an unverified eight-GPU topology.

From the replay checkout, with the Halo checkout at the approved revision and `HALO_DIR` pointing to it, use a Python environment containing `datasets` (verified with 5.2.0) and `PyYAML`:

```sh
python halo_export.py data run/halo-dataset
python halo_export.py config \
  "$HALO_DIR/examples/sft/gemma4/gemma4-26b-a4b-ultrachat-ep.yaml" \
  run/halo-dataset run/halo-smoke.yml run/halo-checkpoint
```

Mount the same dataset and checkpoint paths in the GPU environment, or generate the YAML there to capture its absolute paths. From the installed Halo checkout on a compatible eight-GPU node:

```sh
torchrun --nproc_per_node=8 scripts/training/sft.py /absolute/path/to/run/halo-smoke.yml
```

Smoke pass: both curated splits load as 484/95; exactly two updates run with finite loss; the checkpoint exists and reloads to emit an option letter. Local verification covers exporting and reloading all 579 rows, equality against the MLX inputs, split separation, and YAML generation. The Halo runtime, Gemma 4 tokenizer and checkpoint reload still need a real GPU run. A passing smoke run would validate the port, not prove better compression.
