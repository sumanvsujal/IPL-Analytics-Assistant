"""
IPL Analytics — Dataset Loader & Formatter
============================================
Reads JSONL → formats prompts → tokenizes with label masking.
"""

import json
from pathlib import Path

from datasets import Dataset

from finetuning.configs.training_config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE, format_prompt, MAX_SEQ_LENGTH,
)


# ── I/O ──────────────────────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


# ── Formatting ───────────────────────────────────────────────────────────

def format_samples(samples: list[dict]) -> list[dict]:
    """
    Convert raw JSONL samples to prompt-formatted dicts.

    Returns list of:
      text     — full prompt+output (for training)
      prompt   — prompt only (for inference / label masking)
      expected — raw expected output string
      meta     — original metadata
    """
    out = []
    for s in samples:
        text   = format_prompt(s["instruction"], s.get("input", ""), s["output"])
        prompt = format_prompt(s["instruction"], s.get("input", ""))
        out.append({
            "text":     text,
            "prompt":   prompt,
            "expected": s["output"],
            "meta":     s.get("meta", {}),
        })
    return out


# ── Tokenization ─────────────────────────────────────────────────────────

def tokenize_for_training(samples: list[dict], tokenizer) -> Dataset:
    """
    Tokenize and create labels with prompt masking.

    Labels use -100 for prompt tokens so loss is computed only on
    the output portion the model should learn to generate.
    """
    ds = Dataset.from_list(samples)

    def _tok(example):
        full = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )["input_ids"]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full["input_ids"][prompt_len:]
        full["labels"] = labels[: len(full["input_ids"])]
        return full

    return ds.map(_tok, remove_columns=ds.column_names)


# ── Public entry point ───────────────────────────────────────────────────

def load_and_prepare_datasets(tokenizer, max_train=None, max_val=None):
    """
    Full pipeline:  load JSONL → format → tokenize.

    Args:
        tokenizer:  HuggingFace tokenizer
        max_train:  if set, truncate training set (for smoke tests)
        max_val:    if set, truncate validation set

    Returns:
        (train_dataset, val_dataset, test_formatted_list)
    """
    print("Loading datasets …")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)
    test_raw  = load_jsonl(TEST_FILE)

    if max_train:
        train_raw = train_raw[:max_train]
    if max_val:
        val_raw = val_raw[:max_val]

    print(f"  Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

    print("Formatting …")
    train_fmt = format_samples(train_raw)
    val_fmt   = format_samples(val_raw)
    test_fmt  = format_samples(test_raw)

    print("Tokenizing …")
    train_ds = tokenize_for_training(train_fmt, tokenizer)
    val_ds   = tokenize_for_training(val_fmt, tokenizer)

    print("Done.")
    return train_ds, val_ds, test_fmt
