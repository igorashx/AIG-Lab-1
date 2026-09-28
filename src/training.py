"""Training and validation loops for the character-level language model."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def perplexity(loss: float) -> float:
    """Convert average cross-entropy to perplexity without overflowing."""
    return math.exp(min(loss, 100.0))


def mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress_description: str | None = None,
) -> float:
    """Compute mean token-level loss, weighted by valid target count."""
    model.eval()
    total_loss = 0.0
    token_count = 0
    batches = tqdm(
        loader,
        desc=progress_description,
        unit="batch",
        leave=False,
        disable=progress_description is None,
    )
    with torch.no_grad():
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            valid_count = int((targets != -100).sum().item())
            if valid_count:
                batch_loss = float(criterion(logits.transpose(1, 2), targets).item())
                total_loss += batch_loss * valid_count
                token_count += valid_count
                batches.set_postfix(loss=f"{batch_loss:.4f}")
    if token_count == 0:
        raise ValueError("Loaderul de evaluare nu conține tokenuri țintă valide.")
    return total_loss / token_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict[str, Any],
    show_progress: bool = True,
) -> list[dict[str, float | int]]:
    """Train with validation, checkpointing, and configurable early stopping."""
    if num_epochs < 1 or patience < 1:
        raise ValueError("num_epochs și patience trebuie să fie cel puțin 1.")
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    patience_used = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, num_epochs + 1):
        epoch_started = time.perf_counter()
        epoch_label = f"Epoch {epoch}/{num_epochs}"
        model.train()
        total_loss = 0.0
        token_count = 0
        batches = tqdm(
            train_loader,
            desc=epoch_label,
            unit="batch",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = criterion(logits.transpose(1, 2), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            valid_count = int((targets != -100).sum().item())
            total_loss += float(loss.item()) * valid_count
            token_count += valid_count
            batches.set_postfix(loss=f"{loss.item():.4f}")
        if token_count == 0:
            raise ValueError("Loaderul de antrenare nu conține tokenuri țintă valide.")

        train_loss = total_loss / token_count
        validation_loss = mean_loss(
            model,
            validation_loader,
            criterion,
            device,
            progress_description=f"{epoch_label} | validation" if show_progress else None,
        )
        entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_perplexity": perplexity(train_loss),
            "validation_perplexity": perplexity(validation_loss),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(entry)

        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            patience_used = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
            status = "checkpoint salvat"
        else:
            patience_used += 1
            status = f"fără îmbunătățire ({patience_used}/{patience})"

        print(
            f"{epoch_label} - {entry['epoch_seconds']:.1f}s - "
            f"loss: {train_loss:.4f} - perplexity: {entry['train_perplexity']:.4f} - "
            f"val_loss: {validation_loss:.4f} - val_perplexity: {entry['validation_perplexity']:.4f} - {status}"
        )

        if not improved:
            if patience_used >= patience:
                print(f"Early stopping activat după epoca {epoch}; se păstrează checkpoint-ul optim.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Checkpoint optim încărcat: epoca {checkpoint['epoch']} "
        f"(val_loss={checkpoint['validation_loss']:.4f})."
    )
    return history