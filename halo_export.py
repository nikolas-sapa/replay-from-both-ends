"""Export replay SFT data and adapt Halo's official Gemma example."""

import argparse
import json
from pathlib import Path

from experiment import OPTIONS, prompt, samples


def rows_for(segments):
    for start, end, moves, paths in segments:
        if len(paths) > 1:
            text, options = prompt(start, end, paths)
            yield {"prompt": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": OPTIONS[options.index(moves)]},
            ]}


def export(output):
    from datasets import Dataset, DatasetDict, Features, List, Value, load_from_disk

    if output.exists():
        raise FileExistsError(output)
    splits = samples()
    expected = {"train": list(rows_for(splits["train"])),
                "test": list(rows_for(splits["valid"]))}
    features = Features({"prompt": List({"role": Value("string"), "content": Value("string")})})
    data = DatasetDict({
        name: Dataset.from_list(rows, features=features) for name, rows in expected.items()
    })
    data.save_to_disk(str(output))
    loaded = load_from_disk(str(output))
    if set(loaded) != set(expected) or any(list(loaded[k]) != v for k, v in expected.items()):
        raise ValueError("Saved data does not match source splits")
    print(json.dumps({"train": len(loaded["train"]), "halo_eval": len(loaded["test"])}))


def smoke_config(upstream, dataset, output, checkpoint):
    import yaml

    cfg = yaml.safe_load(upstream.read_text(encoding="utf-8"))
    if cfg.get("model_name_or_path") != "google/gemma-4-26B-A4B-it" or cfg.get("expert_parallel_size") != 8:
        raise ValueError("Use Halo's official Gemma 4 26B-A4B EP8 SFT example")
    if not (dataset / "dataset_dict.json").is_file():
        raise ValueError("Export the DatasetDict first")
    cfg.pop("test_size", None)
    cfg.update(
        dataset=[str(dataset.resolve())],
        conversation_field="prompt",
        train_on_completions_only=True,
        max_length=256,
        packing=False,
        gradient_accumulation_steps=1,
        max_steps=2,
        warmup_steps=0,
        eval_strategy="no",
        save_steps=2,
        output_dir=str(checkpoint.resolve()),
        report_to=[],
    )
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    data.add_argument("output", type=Path)
    config = commands.add_parser("config")
    for field in ("upstream", "dataset", "output", "checkpoint"):
        config.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.command == "data":
        export(args.output)
    else:
        smoke_config(args.upstream, args.dataset, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
