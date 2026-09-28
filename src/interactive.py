"""Interactive terminal session for an already trained character LSTM."""

from __future__ import annotations

import argparse
import secrets
from collections.abc import Callable

import torch

from src.generate import generate_text, load_model_for_inference
from src.model import CharacterLSTM
from src.utils import get_device, load_yaml, set_seed
from src.vocabulary import CharacterVocabulary


def validate_prompt(prompt: str, sequence_unit: str) -> str:
    """Validate one prompt against the sequence unit used during training."""
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Textul inițial nu poate fi gol.")
    if sequence_unit == "sentence" and "." in cleaned_prompt:
        raise ValueError(
            "Modelul a fost antrenat cu sequence_unit='sentence'. "
            "Introdu un prefix fără punct; punctul este urmat de <EOS> la antrenare."
        )
    return cleaned_prompt


def validate_sampling_options(temperature: float, max_length: int) -> None:
    """Validate sampling parameters before beginning an interactive session."""
    if temperature <= 0:
        raise ValueError("Temperatura trebuie să fie strict pozitivă.")
    if max_length < 1:
        raise ValueError("max-length trebuie să fie cel puțin 1.")


def sampling_seed(seed_mode: str, fixed_seed: int) -> int:
    """Return the seed used for one sample and set all supported generators."""
    if seed_mode == "fixed":
        seed = fixed_seed
    elif seed_mode == "random":
        seed = secrets.randbelow(2**32)
    else:
        raise ValueError("seed-mode trebuie să fie 'fixed' sau 'random'.")
    set_seed(seed)
    return seed


def interactive_session(
    model: CharacterLSTM,
    vocabulary: CharacterVocabulary,
    device: torch.device,
    sequence_unit: str,
    temperature: float,
    max_length: int,
    seed_mode: str,
    fixed_seed: int,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Generate repeatedly until the user sends Ctrl+C or end-of-file."""
    validate_sampling_options(temperature, max_length)
    output_fn("\n=== Sesiune interactivă ===")
    output_fn(
        f"temperature={temperature} | max_length={max_length} | "
        f"seed_mode={seed_mode} | sequence_unit={sequence_unit}"
    )
    output_fn("Scrie un prefix și apasă Enter. Ieșire: Ctrl+C sau Ctrl+Z, apoi Enter.")

    while True:
        try:
            prompt = input_fn("\nText inițial > ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nSesiune încheiată.")
            return

        try:
            prompt = validate_prompt(prompt, sequence_unit)
        except ValueError as error:
            output_fn(f"Eroare: {error}")
            continue

        seed = sampling_seed(seed_mode, fixed_seed)
        generated = generate_text(model, vocabulary, prompt, temperature, max_length, device)
        continuation = generated[len(prompt) :]
        termination = "EOS" if len(continuation) < max_length else "limită de siguranță"
        output_fn(f"Seed utilizat: {seed} | caractere generate: {len(continuation)} | oprire: {termination}")
        output_fn("--- Predicție ---")
        output_fn(generated)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the interactive generator."""
    parser = argparse.ArgumentParser(description="Generează repetat text cu un checkpoint LSTM salvat.")
    parser.add_argument("--config", default="config/config.yaml", help="Fișier YAML de configurare.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Checkpoint-ul PyTorch folosit la inferență.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperatura samplingului, strict pozitivă.")
    parser.add_argument("--max-length", type=int, default=None, help="Limita de siguranță în caractere generate.")
    parser.add_argument("--seed-mode", choices=("fixed", "random"), default="fixed", help="fixed repetă același sampling; random alege un seed nou la fiecare încercare.")
    parser.add_argument("--seed", type=int, default=None, help="Seed pentru --seed-mode fixed; implicit valoarea din configurație.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    inference = config["inference"]
    max_length = args.max_length if args.max_length is not None else inference["max_generation_length"]
    fixed_seed = args.seed if args.seed is not None else config["seed"]
    validate_sampling_options(args.temperature, max_length)
    if fixed_seed < 0:
        raise ValueError("Seed-ul trebuie să fie un întreg nenegativ.")

    device = get_device()
    print(f"Se încarcă checkpoint-ul pe {device}...")
    model, vocabulary = load_model_for_inference(config, args.checkpoint, device)
    interactive_session(
        model=model,
        vocabulary=vocabulary,
        device=device,
        sequence_unit=config["data"].get("sequence_unit", "paragraph"),
        temperature=args.temperature,
        max_length=max_length,
        seed_mode=args.seed_mode,
        fixed_seed=fixed_seed,
    )


if __name__ == "__main__":
    main()