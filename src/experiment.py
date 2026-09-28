"""Run training, one final test evaluation, and four temperature samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_test_set
from src.generate import generate_text
from src.reporting import character_diversity, plot_temperature_samples
from src.train import run_training
from src.utils import get_device, load_yaml, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Rulează fluxul complet al laboratorului.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print("\n########################################")
    print("# Experiment LSTM caracter-cu-caracter #")
    print("########################################")
    model, vocabulary, _, history = run_training(config)
    metrics = evaluate_test_set(config)
    device = get_device()
    inference = config["inference"]
    samples = []
    print("\n=== Generare comparativă pe temperaturi ===")
    print(f"Seed text: {inference['seed_text']!r}")
    for temperature in inference["temperatures"]:
        print(f"Generez pentru temperatura T={temperature}...")
        set_seed(config["seed"])
        text = generate_text(model, vocabulary, inference["seed_text"], temperature, inference["max_generation_length"], device)
        generated = text[len(inference["seed_text"]) :]
        samples.append({
            "temperature": temperature,
            "length": len(generated),
            "ended_with_eos": len(generated) < inference["max_generation_length"],
            "text": text,
            **character_diversity(generated),
        })
        ending = "EOS" if len(generated) < inference["max_generation_length"] else "limită de siguranță"
        print(f"T={temperature} - {len(generated)} caractere - oprire: {ending}")
    save_json(samples, Path(config["paths"]["output_dir"]) / "temperature_samples.json")
    plot_temperature_samples(samples, config["paths"].get("plots_dir", Path(config["paths"]["output_dir"]) / "plots"))
    print("\n=== Experiment finalizat ===")
    print(f"Epoci: {len(history)}; PPL test: {metrics['test_perplexity']:.4f}; exemple salvate: {len(samples)}")


if __name__ == "__main__":
    main()