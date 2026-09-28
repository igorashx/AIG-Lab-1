"""Dataset loading, work-level splitting, segmentation, and LSTM windows."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.vocabulary import CharacterVocabulary


@dataclass(frozen=True)
class ParagraphRecord:
    """One paragraph tied to its source literary work."""

    work_id: str
    text: str


def _normalise_key(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _first_available(columns: Iterable[str], candidates: Sequence[str]) -> str:
    normalized = {_normalise_key(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"Nu s-a găsit o coloană compatibilă. Coloane disponibile: {sorted(columns)}")


def records_from_rows(rows: Iterable[dict[str, Any]]) -> list[ParagraphRecord]:
    """Convert dataset rows to records, validating source-work metadata."""
    rows = list(rows)
    if not rows:
        raise ValueError("Datasetul nu conține rânduri.")
    columns = rows[0].keys()
    text_column = _first_available(columns, ("paragraph", "text", "content", "paragraph_text"))
    author_column = _first_available(columns, ("author", "autor", "writer"))
    title_column = _first_available(columns, ("title", "titlu", "work", "book"))

    records: list[ParagraphRecord] = []
    for row in rows:
        text = str(row.get(text_column, "")).strip()
        author = _normalise_key(row.get(author_column, ""))
        title = _normalise_key(row.get(title_column, ""))
        if text and author and title:
            records.append(ParagraphRecord(work_id=f"{author}::{title}", text=text))
    if not records:
        raise ValueError("Nu există paragrafe valide cu autor și titlu.")
    return records


def load_readerbench_records(dataset_name: str) -> list[ParagraphRecord]:
    """Download ReaderBench's published paragraph CSV and extract records."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Instalează dependențele cu `pip install -r requirements.txt`.") from error

    file_path = hf_hub_download(
        repo_id=dataset_name,
        repo_type="dataset",
        filename="ro_paragraphs_12516.csv",
    )
    with open(file_path, encoding="utf-8-sig", newline="") as file:
        return records_from_rows(csv.DictReader(file))


def split_records_by_work(
    records: Sequence[ParagraphRecord],
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[ParagraphRecord]]:
    """Deterministically split complete works; no work crosses subsets."""
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("Fracțiile splitului trebuie să fie strict pozitive.")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Fracțiile splitului trebuie să însumeze 1.")

    grouped: dict[str, list[ParagraphRecord]] = defaultdict(list)
    for record in records:
        grouped[record.work_id].append(record)
    work_ids = list(grouped)
    if len(work_ids) < 3:
        raise ValueError("Sunt necesare minimum trei opere pentru train/validation/test.")

    random.Random(seed).shuffle(work_ids)
    work_count = len(work_ids)
    train_count = min(work_count - 2, max(1, int(work_count * train_fraction)))
    validation_count = min(
        work_count - train_count - 1,
        max(1, int(work_count * validation_fraction)),
    )
    train_end = train_count
    validation_end = train_end + validation_count
    selected = {
        "train": work_ids[:train_end],
        "validation": work_ids[train_end:validation_end],
        "test": work_ids[validation_end:],
    }
    return {subset: [record for work_id in ids for record in grouped[work_id]] for subset, ids in selected.items()}


def split_manifest(splits: dict[str, Sequence[ParagraphRecord]]) -> dict[str, list[str]]:
    return {subset: sorted({record.work_id for record in records}) for subset, records in splits.items()}


def apply_split_manifest(
    records: Sequence[ParagraphRecord], manifest: dict[str, Sequence[str]]
) -> dict[str, list[ParagraphRecord]]:
    assignments = {work_id: subset for subset, work_ids in manifest.items() for work_id in work_ids}
    unexpected = {record.work_id for record in records} - set(assignments)
    if unexpected:
        raise ValueError("Datasetul nu corespunde manifestului salvat al splitului.")
    return {subset: [record for record in records if assignments[record.work_id] == subset] for subset in manifest}


def split_text_into_sequences(text: str, sequence_unit: str) -> list[str]:
    """Split one paragraph into independent training sequences.

    Sentence mode uses every literal full stop as a boundary, retaining the full
    stop in the preceding sequence. A final non-empty fragment is retained so
    that no corpus text is discarded.
    """
    if sequence_unit == "paragraph":
        return [text] if text.strip() else []
    if sequence_unit != "sentence":
        raise ValueError("sequence_unit trebuie să fie 'paragraph' sau 'sentence'.")

    sequences: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character == ".":
            sequence = text[start : index + 1].strip()
            if sequence:
                sequences.append(sequence)
            start = index + 1
    final_sequence = text[start:].strip()
    if final_sequence:
        sequences.append(final_sequence)
    return sequences


def iter_sequence_texts(records: Iterable[ParagraphRecord], sequence_unit: str) -> Iterable[str]:
    """Yield paragraph or full-stop-delimited sentence sequences per record."""
    for record in records:
        yield from split_text_into_sequences(record.text, sequence_unit)


class CharacterWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create next-token windows independently for each sequence plus EOS."""

    def __init__(
        self,
        texts: Iterable[str],
        vocabulary: CharacterVocabulary,
        sequence_length: int,
        stride: int | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length trebuie să fie cel puțin 1.")
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        stride = stride or sequence_length
        if stride < 1:
            raise ValueError("stride trebuie să fie cel puțin 1.")
        for text in texts:
            token_ids = vocabulary.encode(text, append_eos=True)
            for start in range(0, len(token_ids) - 1, stride):
                chunk = token_ids[start : start + sequence_length + 1]
                if len(chunk) >= 2:
                    self.examples.append(
                        (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long))
                    )
        if not self.examples:
            raise ValueError("Nu au rezultat exemple. Verifică paragrafele și sequence_length.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_windows(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]], pad_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return padded_inputs, padded_targets