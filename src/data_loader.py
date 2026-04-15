"""
src/data_loader.py
------------------
Dataset ingestion and preprocessing module for InstructTune-LLM.

Responsibilities
----------------
1. Load raw JSON / JSONL records from a local file or the HuggingFace Hub
2. Validate that every record satisfies the required three-field schema
3. Apply PromptComposer to format records into model-ready strings
4. Tokenise each string with optional response-only loss masking
5. Partition the processed dataset into train and validation subsets

Design philosophy: all parameters are provided at construction time.
The public entry point `prepare_dataset()` is intentionally stateless
between calls — you can call it multiple times with different configs.
"""

from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.prompt_builder import PromptComposer

_log = logging.getLogger(__name__)

# Every raw record must carry these three keys (or their aliases)
SCHEMA_FIELDS = {"task", "context", "response"}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _check_record_schema(record: dict, row_idx: int) -> None:
    """Raise a descriptive KeyError when required fields are absent."""
    missing = SCHEMA_FIELDS - set(record.keys())
    if missing:
        raise KeyError(
            f"Row {row_idx} is missing required fields: {missing}. "
            f"Found fields: {list(record.keys())}"
        )


# ---------------------------------------------------------------------------
# RecordIngester
# ---------------------------------------------------------------------------

class RecordIngester:
    """
    Handles the complete data pipeline from raw JSON on disk to a pair
    of tokenised HuggingFace Datasets consumable by the Trainer.

    Parameters
    ----------
    file_path : str
        Local path to a .json / .jsonl file, or a HuggingFace Hub dataset ID.
    tokenizer : PreTrainedTokenizerBase
        Already-configured tokenizer for the target backbone model.
    composer : PromptComposer
        Initialised PromptComposer used to format raw records.
    column_map : dict, optional
        Maps the canonical schema keys (task / context / response) to the
        actual column names in the raw data.  Defaults to identity mapping.
    max_seq_len : int
        Hard token limit per sample; longer sequences are truncated.
    val_set_size : int
        Number of rows reserved for validation.  0 disables validation.
    response_only_loss : bool
        When True, cross-entropy labels for prompt tokens are set to -100
        (masked out), so the model only learns from its own responses.
    append_eos_token : bool
        Append the EOS token after each tokenised sequence if not already
        present.  Important for autoregressive generation quality.
    shuffle_seed : int
        Seed for reproducible shuffling and train/val splitting.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        composer: PromptComposer,
        column_map: Optional[dict[str, str]] = None,
        max_seq_len: int = 512,
        val_set_size: int = 200,
        response_only_loss: bool = True,
        append_eos_token: bool = True,
        shuffle_seed: int = 42,
    ) -> None:
        self._path = file_path
        self._tok = tokenizer
        self._composer = composer
        self._col_map = column_map or {
            "task": "task",
            "context": "context",
            "response": "response",
        }
        self._max_len = max_seq_len
        self._val_size = val_set_size
        self._resp_only = response_only_loss
        self._append_eos = append_eos_token
        self._seed = shuffle_seed

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        """
        Execute the full pipeline: load → validate → format → tokenise → split.

        Returns
        -------
        tuple[Dataset, Dataset | None]
            (train_split, val_split) — val_split is None when val_set_size=0.
        """
        _log.info("Ingesting records from: '%s'", self._path)
        raw = self._load_raw()

        total_rows = len(raw["train"])
        _log.info("Tokenising %d records …", total_rows)

        # Shuffle first to randomise the train/val boundary
        shuffled = raw["train"].shuffle(seed=self._seed)
        processed = shuffled.map(
            self._process_row,
            remove_columns=shuffled.column_names,
        )

        if self._val_size > 0:
            partitioned = processed.train_test_split(
                test_size=self._val_size,
                shuffle=True,
                seed=self._seed,
            )
            train_data = partitioned["train"]
            val_data = partitioned["test"]
            _log.info(
                "Split: %d training rows, %d validation rows",
                len(train_data),
                len(val_data),
            )
        else:
            train_data = processed
            val_data = None
            _log.info(
                "No validation split — using all %d rows for training.",
                len(train_data),
            )

        return train_data, val_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> DatasetDict:
        """Read raw data from a local JSON/JSONL file or the HF Hub."""
        src = self._path
        if src.endswith(".json") or src.endswith(".jsonl"):
            return load_dataset("json", data_files=src)
        return load_dataset(src)

    def _tokenize_text(self, text: str, with_eos: bool = True) -> dict:
        """
        Tokenise a single string and optionally append the EOS token.

        The tokenizer is called with padding=False so sequences keep their
        natural lengths — the DataCollator handles batch-level padding.
        """
        encoded = self._tok(
            text,
            truncation=True,
            max_length=self._max_len,
            padding=False,
            return_tensors=None,
        )
        if (
            with_eos
            and encoded["input_ids"][-1] != self._tok.eos_token_id
            and len(encoded["input_ids"]) < self._max_len
        ):
            encoded["input_ids"].append(self._tok.eos_token_id)
            encoded["attention_mask"].append(1)

        # Labels start as a copy of input_ids; masking happens below
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    def _process_row(self, row: dict) -> dict:
        """
        Convert one raw JSON record into a tokenised training example.

        When response_only_loss=True the label positions that correspond to
        prompt tokens are replaced with -100 so they do not contribute to
        the cross-entropy loss, forcing the model to learn only responses.
        """
        task_text = row.get(self._col_map["task"], "")
        ctx_text = row.get(self._col_map["context"], "") or None
        answer_text = row.get(self._col_map["response"], "")

        # Full sequence: prompt + target response (used as input during training)
        full_prompt = self._composer.build_prompt(
            task=task_text,
            context=ctx_text,
            target_output=answer_text,
        )
        tokenised = self._tokenize_text(full_prompt, with_eos=self._append_eos)

        if not self._resp_only:
            return tokenised

        # Compute prompt-only length to determine the masking boundary
        prompt_prefix = self._composer.build_prompt(
            task=task_text,
            context=ctx_text,
        )
        prompt_tokens = self._tokenize_text(prompt_prefix, with_eos=False)
        prefix_len = len(prompt_tokens["input_ids"])

        # Mask all prompt positions so loss ignores them
        tokenised["labels"] = (
            [-100] * prefix_len + tokenised["labels"][prefix_len:]
        )
        return tokenised
