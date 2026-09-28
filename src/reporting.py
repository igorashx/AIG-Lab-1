"""Metrics summaries and non-interactive plots for laboratory experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn

from src.dataset import ParagraphRecord
from src.utils import save_json
from src.vocabulary import CharacterVocabulary


def split_statistics(
    splits: dict[str, Sequence[ParagraphRecord]], vocabulary: CharacterVocabulary
) -> dict[str, dict[str, float | int]]:
    """Return corpus size and OOV statistics for each work-level split."""
    summary: dict[str, dict[str, float | int]] = {}
    for subset, records in splits.items():
        texts = [record.text for record in records]
        character_count = sum(len(text) for text in texts)
        unknown_count = sum(
            character not in vocabulary.char_to_idx for text in texts for character in text
        )
        summary[subset] = {
            "works": len({record.work_id for record in records}),
            "paragraphs": len(records),
            "characters": character_count,
            "mean_paragraph_length": character_count / len(texts) if texts else 0.0,
            "unknown_characters": unknown_count,
            "unknown_character_rate": unknown_count / character_count if character_count else 0.0,
        }
    return summary


def model_statistics(model: nn.Module, vocabulary: CharacterVocabulary, device: str) -> dict[str, int | str]:
    """Summarize model capacity and runtime context."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "vocabulary_size": vocabulary.size,
        "parameters_total": parameter_count,
        "parameters_trainable": trainable_count,
        "device": device,
    }


def save_training_summary(
    history: Sequence[dict[str, float | int]],
    split_metrics: dict[str, dict[str, float | int]],
    model_metrics: dict[str, int | str],
    path: str | Path,
) -> dict:
    """Persist an easy-to-read summary without accessing the test set."""
    best = min(history, key=lambda row: float(row["validation_loss"]))
    summary = {
        "completed_epochs": len(history),
        "best_epoch": best["epoch"],
        "best_validation_loss": best["validation_loss"],
        "best_validation_perplexity": best["validation_perplexity"],
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "total_training_seconds": sum(float(row.get("epoch_seconds", 0.0)) for row in history),
        "split": split_metrics,
        "model": model_metrics,
    }
    save_json(summary, path)
    return summary


def plot_training_history(history: Sequence[dict[str, float | int]], output_dir: str | Path) -> list[Path]:
    """Save loss and perplexity curves for train and validation."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    plots: list[Path] = []
    for metric, label, filename in (
        ("loss", "Cross-entropy loss", "training_loss.png"),
        ("perplexity", "Perplexity", "training_perplexity.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, [float(row[f"train_{metric}"]) for row in history], marker="o", label="Train")
        axis.plot(epochs, [float(row[f"validation_{metric}"]) for row in history], marker="o", label="Validation")
        axis.set(title=f"{label} pe epocă", xlabel="Epocă", ylabel=label)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        path = directory / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        plots.append(path)
    return plots


def plot_final_evaluation(metrics: dict[str, float], output_dir: str | Path) -> Path:
    """Save the final held-out test loss and perplexity as separate bars."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, key, label in zip(axes, ("test_loss", "test_perplexity"), ("Test loss", "Test perplexity")):
        axis.bar(["Test"], [metrics[key]], color="tab:blue")
        axis.set(title=label, ylabel=label)
    figure.suptitle("Performanța checkpoint-ului selectat")
    figure.tight_layout()
    path = directory / "final_test_metrics.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_split_statistics(metrics: dict[str, dict[str, float | int]], output_dir: str | Path) -> Path:
    """Save a chart of works, paragraphs and characters per split."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    subsets = list(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, key, label in zip(axes, ("works", "paragraphs", "characters"), ("Opere", "Paragrafe", "Caractere")):
        axis.bar(subsets, [float(metrics[subset][key]) for subset in subsets])
        axis.set(title=label, ylabel="Număr")
    figure.suptitle("Dimensiunea subseturilor")
    figure.tight_layout()
    path = directory / "split_statistics.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_temperature_samples(samples: Iterable[dict], output_dir: str | Path) -> Path:
    """Save generated lengths and termination status by temperature."""
    rows = list(samples)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    labels = [str(row["temperature"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    colors = ["tab:green" if row.get("ended_with_eos") else "tab:orange" for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, lengths, color=colors)
    axis.set(title="Lungimea generată până la EOS", xlabel="Temperatură", ylabel="Caractere generate")
    figure.tight_layout()
    path = directory / "temperature_lengths.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def character_diversity(text: str) -> dict[str, float | int]:
    """Compute lightweight descriptive metrics for a generated string."""
    if not text:
        return {"unique_characters": 0, "unique_character_ratio": 0.0, "repeated_adjacent_pairs": 0}
    pairs = Counter(zip(text, text[1:]))
    return {
        "unique_characters": len(set(text)),
        "unique_character_ratio": len(set(text)) / len(text),
        "repeated_adjacent_pairs": sum(count - 1 for count in pairs.values() if count > 1),
    }