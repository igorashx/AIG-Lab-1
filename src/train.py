"""Train the LSTM character language model on readerbench/ro-stories."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import CharacterWindowDataset, collate_windows, iter_sequence_texts, load_readerbench_records, split_manifest, split_records_by_work
from src.model import CharacterLSTM
from src.reporting import model_statistics, plot_split_statistics, plot_training_history, save_training_summary, split_statistics
from src.training import train_model
from src.utils import get_device, load_yaml, make_torch_generator, save_json, save_yaml, seed_data_loader_worker, set_seed
from src.vocabulary import CharacterVocabulary


def build_loaders(config: dict, splits: dict[str, list], vocabulary: CharacterVocabulary) -> dict[str, DataLoader]:
    data_config = config["data"]
    training_config = config["training"]
    datasets = {
        subset: CharacterWindowDataset(
            iter_sequence_texts(records, data_config.get("sequence_unit", "paragraph")),
            vocabulary,
            data_config["sequence_length"],
            data_config["window_stride"],
        )
        for subset, records in splits.items()
    }
    collate = partial(collate_windows, pad_idx=vocabulary.pad_idx)
    loader_seeds = {"train": config["seed"], "validation": config["seed"] + 1, "test": config["seed"] + 2}
    return {
        subset: DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=subset == "train",
            num_workers=training_config["num_workers"],
            collate_fn=collate,
            generator=make_torch_generator(loader_seeds[subset]),
            worker_init_fn=seed_data_loader_worker,
        )
        for subset, dataset in datasets.items()
    }


def run_training(config: dict) -> tuple[CharacterLSTM, CharacterVocabulary, dict[str, DataLoader], list[dict]]:
    """Run the full train/validation workflow and persist all development artifacts."""
    set_seed(config["seed"])
    device = get_device()
    print("\n=== Pregătirea experimentului ===")
    print(f"Seed fixat: {config['seed']} | mod determinist: activ | dispozitiv: {device}")
    print(f"Se încarcă datasetul: {config['data']['dataset_name']}...")
    records = load_readerbench_records(config["data"]["dataset_name"])
    print(f"Dataset încărcat: {len(records):,} paragrafe.")
    sequence_unit = config["data"].get("sequence_unit", "paragraph")
    print(f"Unitate de secvență: {sequence_unit} (fiecare secvență primește un token <EOS>).")
    split_config = config["data"]["split"]
    splits = split_records_by_work(records, config["seed"], **{
        "train_fraction": split_config["train"],
        "validation_fraction": split_config["validation"],
        "test_fraction": split_config["test"],
    })
    print(
        "Split pe opere: "
        + ", ".join(f"{name}={len({record.work_id for record in subset})}" for name, subset in splits.items())
    )
    vocabulary = CharacterVocabulary.build(record.text for record in splits["train"])
    print(f"Vocabular construit exclusiv din train: {vocabulary.size} tokenuri.")
    vocabulary.save(config["paths"]["vocabulary_file"])
    save_json(split_manifest(splits), config["paths"]["split_file"])
    save_yaml(config, config["paths"]["config_file"])
    loaders = build_loaders(config, splits, vocabulary)
    print(
        "Ferestre LSTM: "
        + ", ".join(f"{name}={len(loader.dataset):,}" for name, loader in loaders.items())
    )

    model = CharacterLSTM(vocab_size=vocabulary.size, pad_idx=vocabulary.pad_idx, **config["model"]).to(device)
    training_config = config["training"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model LSTM inițializat: {parameter_count:,} parametri antrenabili.")
    print("\n=== Antrenare ===")
    optimizer = Adam(model.parameters(), lr=training_config["learning_rate"])
    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
    history = train_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        optimizer=optimizer,
        device=device,
        num_epochs=training_config["num_epochs"],
        patience=training_config["patience"],
        gradient_clip_norm=training_config["gradient_clip_norm"],
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={
            "vocab_size": vocabulary.size,
            "model_config": config["model"],
            "sequence_unit": sequence_unit,
        },
        show_progress=training_config.get("show_progress", True),
    )
    save_json(history, config["paths"]["history_file"])
    metrics = split_statistics(splits, vocabulary)
    output_dir = Path(config["paths"]["output_dir"])
    plots_dir = config["paths"].get("plots_dir", output_dir / "plots")
    save_training_summary(
        history,
        metrics,
        model_statistics(model, vocabulary, str(device)),
        config["paths"].get("training_summary_file", output_dir / "training_summary.json"),
    )
    plot_training_history(history, plots_dir)
    plot_split_statistics(metrics, plots_dir)
    print(f"Artefacte și grafice salvate în: {output_dir}")
    return model, vocabulary, loaders, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Antrenează generatorul LSTM caracter-cu-caracter.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    _, vocabulary, loaders, history = run_training(config)
    print(f"Antrenare finalizată: {len(history)} epoci, vocabular={vocabulary.size}, ferestre test={len(loaders['test'].dataset)}")


if __name__ == "__main__":
    main()