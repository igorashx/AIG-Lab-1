from __future__ import annotations

from functools import partial
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import CharacterWindowDataset, ParagraphRecord, collate_windows
from src.evaluate import evaluate_test_set
from src.generate import generate_text
from src.model import CharacterLSTM
from src.reporting import plot_final_evaluation, plot_split_statistics, plot_temperature_samples, plot_training_history, save_training_summary, split_statistics
from src.train import build_loaders, run_training
from src.training import train_model
from src.utils import set_seed
from src.vocabulary import CharacterVocabulary


def _model(vocabulary: CharacterVocabulary) -> CharacterLSTM:
    return CharacterLSTM(
        vocab_size=vocabulary.size,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=1,
        dropout=0.0,
        pad_idx=vocabulary.pad_idx,
    )


def test_model_output_shape() -> None:
    vocabulary = CharacterVocabulary.build(["abc"])
    model = _model(vocabulary)
    logits, hidden = model(torch.tensor([[3, 4, 5]], dtype=torch.long))
    assert logits.shape == (1, 3, vocabulary.size)
    assert hidden[0].shape == (1, 1, 6)


def test_set_seed_repeats_python_numpy_and_torch_sequences() -> None:
    set_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    set_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert first == second


def test_train_loader_shuffle_is_repeatable() -> None:
    vocabulary = CharacterVocabulary.build(["abcdef"])
    records = {subset: [ParagraphRecord(f"a::{subset}", "abcdef")] for subset in ("train", "validation", "test")}
    config = {
        "seed": 19,
        "data": {"sequence_length": 1, "window_stride": 1},
        "training": {"batch_size": 1, "num_workers": 0},
    }
    first_loader = build_loaders(config, records, vocabulary)["train"]
    second_loader = build_loaders(config, records, vocabulary)["train"]
    first_order = [int(inputs[0, 0]) for inputs, _ in first_loader]
    second_order = [int(inputs[0, 0]) for inputs, _ in second_loader]
    assert first_order == second_order


def test_training_saves_and_reloads_best_checkpoint(tmp_path) -> None:
    torch.manual_seed(3)
    vocabulary = CharacterVocabulary.build(["abcd"])
    dataset = CharacterWindowDataset(["abcd"], vocabulary, sequence_length=8)
    loader = DataLoader(dataset, batch_size=1, collate_fn=partial(collate_windows, pad_idx=vocabulary.pad_idx))
    model = _model(vocabulary)
    history = train_model(
        model, loader, loader, Adam(model.parameters(), lr=0.01), torch.device("cpu"),
        num_epochs=2, patience=2, gradient_clip_norm=1.0,
        checkpoint_path=tmp_path / "best_model.pth", checkpoint_metadata={"vocab_size": vocabulary.size},
    )
    assert (tmp_path / "best_model.pth").exists()
    assert len(history) >= 1
    assert all("validation_perplexity" in item for item in history)


def test_generation_stops_when_eos_is_sampled() -> None:
    vocabulary = CharacterVocabulary.build(["ab"])
    model = _model(vocabulary)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias[vocabulary.eos_idx] = 100.0
    generated = generate_text(model, vocabulary, "a", temperature=0.8, max_generation_length=10, device=torch.device("cpu"))
    assert generated == "a"


def test_generation_never_emits_internal_unk_token() -> None:
    vocabulary = CharacterVocabulary.build(["ab"])
    model = _model(vocabulary)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias[vocabulary.unk_idx] = 100.0
        model.output.bias[vocabulary.eos_idx] = 99.0
    generated = generate_text(model, vocabulary, "a", temperature=1.0, max_generation_length=10, device=torch.device("cpu"))
    assert generated == "a"


def test_offline_training_evaluation_and_generation_pipeline(tmp_path, monkeypatch) -> None:
    records = [ParagraphRecord(f"autor::opera-{index}", "ab") for index in range(5)]
    monkeypatch.setattr("src.train.load_readerbench_records", lambda _: records)
    monkeypatch.setattr("src.evaluate.load_readerbench_records", lambda _: records)
    config = {
        "seed": 7,
        "data": {
            "dataset_name": "offline-test",
            "sequence_length": 4,
            "window_stride": 4,
            "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
        },
        "model": {"embedding_dim": 4, "hidden_dim": 6, "num_layers": 1, "dropout": 0.0},
        "training": {
            "batch_size": 2,
            "learning_rate": 0.01,
            "num_epochs": 1,
            "patience": 1,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
        },
        "paths": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "output_dir": str(tmp_path / "outputs"),
            "vocabulary_file": str(tmp_path / "outputs" / "vocabulary.json"),
            "history_file": str(tmp_path / "outputs" / "history.json"),
            "split_file": str(tmp_path / "outputs" / "split.json"),
            "config_file": str(tmp_path / "outputs" / "config.yaml"),
        },
    }
    model, vocabulary, _, history = run_training(config)
    metrics = evaluate_test_set(config)
    generated = generate_text(model, vocabulary, "a", temperature=0.8, max_generation_length=3, device=torch.device("cpu"))

    assert len(history) == 1
    assert (tmp_path / "checkpoints" / "best_model.pth").exists()
    assert metrics["test_loss"] >= 0
    assert generated.startswith("a")


def test_reporting_saves_summaries_and_plots(tmp_path) -> None:
    vocabulary = CharacterVocabulary.build(["ab"])
    splits = {
        "train": [ParagraphRecord("a::1", "ab")],
        "validation": [ParagraphRecord("a::2", "ba")],
        "test": [ParagraphRecord("a::3", "az")],
    }
    history = [{
        "epoch": 1,
        "train_loss": 1.5,
        "validation_loss": 1.7,
        "train_perplexity": 4.5,
        "validation_perplexity": 5.5,
    }]
    metrics = split_statistics(splits, vocabulary)
    summary = save_training_summary(history, metrics, {"parameters_total": 10}, tmp_path / "summary.json")
    history_plots = plot_training_history(history, tmp_path / "plots")
    split_plot = plot_split_statistics(metrics, tmp_path / "plots")
    evaluation_plot = plot_final_evaluation({"test_loss": 1.2, "test_perplexity": 3.3}, tmp_path / "plots")
    temperature_plot = plot_temperature_samples(
        [{"temperature": 0.8, "length": 4, "ended_with_eos": True}], tmp_path / "plots"
    )

    assert summary["best_epoch"] == 1
    assert metrics["test"]["unknown_characters"] == 1
    assert (tmp_path / "summary.json").exists()
    assert all(path.exists() for path in [*history_plots, split_plot, evaluation_plot, temperature_plot])