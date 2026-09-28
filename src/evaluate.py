"""Final test-set evaluation for a saved best checkpoint."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CharacterWindowDataset, apply_split_manifest, collate_windows, iter_sequence_texts, load_readerbench_records
from src.generate import load_model_for_inference
from src.reporting import plot_final_evaluation
from src.training import mean_loss, perplexity
from src.utils import get_device, load_json, load_yaml, save_json


def evaluate_test_set(config: dict, checkpoint_path: str | Path | None = None) -> dict[str, float]:
    """Evaluate only the held-out test split saved during training."""
    device = get_device()
    print("\n=== Evaluare finală pe test ===")
    print("Se încarcă manifestul splitului și checkpoint-ul optim...")
    records = load_readerbench_records(config["data"]["dataset_name"])
    splits = apply_split_manifest(records, load_json(config["paths"]["split_file"]))
    checkpoint_path = checkpoint_path or Path(config["paths"]["checkpoint_dir"]) / "best_model.pth"
    model, vocabulary = load_model_for_inference(config, str(checkpoint_path), device)
    data_config = config["data"]
    test_dataset = CharacterWindowDataset(
        iter_sequence_texts(splits["test"], data_config.get("sequence_unit", "paragraph")),
        vocabulary,
        data_config["sequence_length"],
        data_config["window_stride"],
    )
    print(f"Se evaluează {len(test_dataset):,} ferestre de test pe {device}...")
    loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=partial(collate_windows, pad_idx=vocabulary.pad_idx),
    )
    test_loss = mean_loss(
        model,
        loader,
        nn.CrossEntropyLoss(ignore_index=-100),
        device,
        progress_description="Test",
    )
    metrics = {"test_loss": test_loss, "test_perplexity": perplexity(test_loss)}
    save_json(metrics, Path(config["paths"]["output_dir"]) / "test_metrics.json")
    plot_final_evaluation(metrics, config["paths"].get("plots_dir", Path(config["paths"]["output_dir"]) / "plots"))
    print(f"Test final - loss: {test_loss:.4f} - perplexity: {metrics['test_perplexity']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluează checkpoint-ul optim o singură dată pe test.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    metrics = evaluate_test_set(load_yaml(args.config), args.checkpoint)
    print(f"Test loss: {metrics['test_loss']:.4f}; test perplexity: {metrics['test_perplexity']:.4f}")


if __name__ == "__main__":
    main()