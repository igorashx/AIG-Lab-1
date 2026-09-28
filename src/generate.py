"""Autoregressive EOS-aware temperature sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import CharacterLSTM
from src.utils import get_device, load_yaml
from src.vocabulary import CharacterVocabulary, PAD_TOKEN, UNK_TOKEN


@torch.no_grad()
def generate_text(
    model: CharacterLSTM,
    vocabulary: CharacterVocabulary,
    seed_text: str,
    temperature: float,
    max_generation_length: int,
    device: torch.device,
) -> str:
    """Sample characters until EOS or the explicit safety limit."""
    if temperature <= 0:
        raise ValueError("temperature trebuie să fie strict pozitivă.")
    if max_generation_length < 1:
        raise ValueError("max_generation_length trebuie să fie cel puțin 1.")
    model.eval()
    generated = seed_text
    context = vocabulary.encode(seed_text) or [vocabulary.unk_idx]
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    logits, hidden = model(input_ids)
    next_logits = logits[:, -1, :]

    for _ in range(max_generation_length):
        sampling_logits = next_logits.clone()
        sampling_logits[:, vocabulary.pad_idx] = float("-inf")
        sampling_logits[:, vocabulary.unk_idx] = float("-inf")
        probabilities = torch.softmax(sampling_logits / temperature, dim=-1)
        token_id = int(torch.multinomial(probabilities, num_samples=1).item())
        if token_id == vocabulary.eos_idx:
            break
        token = vocabulary.idx_to_char[token_id]
        if token not in (PAD_TOKEN, UNK_TOKEN):
            generated += token
        input_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, hidden = model(input_token, hidden)
        next_logits = logits[:, -1, :]
    return generated


def load_model_for_inference(config: dict, checkpoint_path: str, device: torch.device) -> tuple[CharacterLSTM, CharacterVocabulary]:
    vocabulary = CharacterVocabulary.load(config["paths"]["vocabulary_file"])
    model = CharacterLSTM(vocab_size=vocabulary.size, pad_idx=vocabulary.pad_idx, **config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_unit = config["data"].get("sequence_unit", "paragraph")
    checkpoint_unit = checkpoint.get("sequence_unit", "paragraph")
    if checkpoint_unit != expected_unit:
        raise ValueError(
            "Checkpoint-ul a fost antrenat cu sequence_unit="
            f"{checkpoint_unit!r}, dar configurația cere {expected_unit!r}. "
            "Rulează din nou antrenarea pentru a crea un checkpoint compatibil."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generează text cu checkpoint-ul LSTM.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--seed-text", default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    device = get_device()
    print(f"Se încarcă checkpoint-ul pe {device}...")
    model, vocabulary = load_model_for_inference(config, args.checkpoint, device)
    inference = config["inference"]
    generated = generate_text(
        model, vocabulary, args.seed_text or inference["seed_text"], args.temperature,
        args.max_length or inference["max_generation_length"], device,
    )
    print(f"Generare finalizată: {len(generated) - len(args.seed_text or inference['seed_text'])} caractere produse.")
    print("\n--- Text generat ---")
    print(generated)


if __name__ == "__main__":
    main()