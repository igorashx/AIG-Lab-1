from __future__ import annotations

import pytest
import torch

from src.interactive import build_parser, interactive_session, sampling_seed, validate_prompt, validate_sampling_options


def test_sentence_prompt_requires_non_empty_text_without_full_stop() -> None:
    assert validate_prompt("  Într-o zi  ", "sentence") == "Într-o zi"
    with pytest.raises(ValueError, match="nu poate fi gol"):
        validate_prompt("   ", "sentence")
    with pytest.raises(ValueError, match="fără punct"):
        validate_prompt("Într-o zi.", "sentence")
    assert validate_prompt("Text. acceptat", "paragraph") == "Text. acceptat"


def test_sampling_options_are_validated() -> None:
    validate_sampling_options(0.8, 20)
    with pytest.raises(ValueError, match="Temperatura"):
        validate_sampling_options(0.0, 20)
    with pytest.raises(ValueError, match="max-length"):
        validate_sampling_options(0.8, 0)


def test_sampling_seed_fixed_and_random(monkeypatch) -> None:
    assert sampling_seed("fixed", 123) == 123
    limits: list[int] = []

    def fake_randbelow(limit: int) -> int:
        limits.append(limit)
        return 456

    monkeypatch.setattr("src.interactive.secrets.randbelow", fake_randbelow)
    assert sampling_seed("random", 123) == 456
    assert limits == [2**32]


def test_interactive_session_handles_invalid_input_and_eof(monkeypatch) -> None:
    responses = iter(["", "Prefix", EOFError()])
    output: list[str] = []
    calls: list[tuple[str, float, int]] = []

    def fake_input(_: str) -> str:
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    def fake_generate(model, vocabulary, prompt, temperature, max_length, device):
        calls.append((prompt, temperature, max_length))
        return f"{prompt} continuare"

    monkeypatch.setattr("src.interactive.generate_text", fake_generate)
    interactive_session(
        model=None,
        vocabulary=None,
        device=torch.device("cpu"),
        sequence_unit="sentence",
        temperature=0.8,
        max_length=10,
        seed_mode="fixed",
        fixed_seed=42,
        input_fn=fake_input,
        output_fn=output.append,
    )

    assert calls == [("Prefix", 0.8, 10)]
    assert any("nu poate fi gol" in line for line in output)
    assert any("Seed utilizat: 42" in line for line in output)
    assert output[-1] == "\nSesiune încheiată."


def test_interactive_session_handles_keyboard_interrupt() -> None:
    output: list[str] = []

    def interrupted_input(_: str) -> str:
        raise KeyboardInterrupt

    interactive_session(
        model=None,
        vocabulary=None,
        device=torch.device("cpu"),
        sequence_unit="sentence",
        temperature=0.8,
        max_length=10,
        seed_mode="fixed",
        fixed_seed=42,
        input_fn=interrupted_input,
        output_fn=output.append,
    )
    assert output[-1] == "\nSesiune încheiată."


def test_parser_exposes_sampling_controls() -> None:
    args = build_parser().parse_args([
        "--temperature", "1.2", "--max-length", "123", "--seed-mode", "random",
    ])
    assert (args.temperature, args.max_length, args.seed_mode) == (1.2, 123, "random")