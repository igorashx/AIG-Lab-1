from __future__ import annotations

import torch

from src.dataset import CharacterWindowDataset, ParagraphRecord, collate_windows, split_records_by_work, split_text_into_sequences
from src.vocabulary import EOS_TOKEN, CharacterVocabulary


def test_vocabulary_is_train_only_and_appends_single_eos() -> None:
    vocabulary = CharacterVocabulary.build(["abă"])
    assert vocabulary.encode("z") == [vocabulary.unk_idx]
    assert vocabulary.encode("ab", append_eos=True)[-1] == vocabulary.eos_idx
    assert vocabulary.idx_to_char[vocabulary.eos_idx] == EOS_TOKEN


def test_work_split_is_disjoint_and_deterministic() -> None:
    records = [
        ParagraphRecord("author::work-1", "unu"),
        ParagraphRecord("author::work-1", "doi"),
        ParagraphRecord("author::work-2", "trei"),
        ParagraphRecord("author::work-3", "patru"),
        ParagraphRecord("author::work-4", "cinci"),
        ParagraphRecord("author::work-5", "șase"),
    ]
    first = split_records_by_work(records, seed=11)
    second = split_records_by_work(records, seed=11)
    first_work_sets = {name: {record.work_id for record in values} for name, values in first.items()}
    assert first_work_sets == {name: {record.work_id for record in values} for name, values in second.items()}
    assert all(first_work_sets.values())
    assert not (first_work_sets["train"] & first_work_sets["validation"])
    assert not (first_work_sets["train"] & first_work_sets["test"])
    assert not (first_work_sets["validation"] & first_work_sets["test"])


def test_windows_preserve_sequence_boundary_and_eos_target() -> None:
    vocabulary = CharacterVocabulary.build(["ab", "cd"])
    dataset = CharacterWindowDataset(["ab", "cd"], vocabulary, sequence_length=8)
    targets = [target.tolist() for _, target in dataset]
    assert targets == [[vocabulary.char_to_idx["b"], vocabulary.eos_idx], [vocabulary.char_to_idx["d"], vocabulary.eos_idx]]
    inputs, padded_targets = collate_windows([dataset[0], dataset[1]], vocabulary.pad_idx)
    assert inputs.shape == padded_targets.shape == torch.Size([2, 2])


def test_sentence_segmentation_keeps_full_stop_and_eos_per_sentence() -> None:
    sequences = split_text_into_sequences("Ana pleacă.  Dan rămâne. Final fără punct", "sentence")
    assert sequences == ["Ana pleacă.", "Dan rămâne.", "Final fără punct"]

    vocabulary = CharacterVocabulary.build(sequences)
    dataset = CharacterWindowDataset(sequences, vocabulary, sequence_length=64)
    assert len(dataset) == 3
    assert all(target[-1].item() == vocabulary.eos_idx for _, target in dataset)
    assert vocabulary.char_to_idx["."] in dataset[0][0].tolist()