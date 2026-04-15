"""
src/prompt_builder.py
---------------------
Prompt composition engine for InstructTune-LLM.

Every input to the model — during both training and generation — is
assembled here.  The FORMAT_CATALOG dictionary registers all supported
prompt layouts.  Swapping layouts requires only a config change; no
code needs to touch this file again.

Each catalog entry defines:
    full_with_bg  : template string used when background context is available
    full_no_bg    : template string when no background is provided
    cutoff_marker : literal string that separates prompt prefix from response;
                    used at decode time to isolate the generated portion
"""

from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Format Catalog
# ---------------------------------------------------------------------------

FORMAT_CATALOG: dict[str, dict[str, str]] = {

    # ── Primary format used in training and default inference ──
    "domain_instruct": {
        "full_with_bg": (
            "### Task:\n{task}\n\n"
            "### Context:\n{context}\n\n"
            "### Response:\n"
        ),
        "full_no_bg": (
            "### Task:\n{task}\n\n"
            "### Response:\n"
        ),
        "cutoff_marker": "### Response:",
    },

    # ── Bracket-style for conversational assistant datasets ──
    "chat_style": {
        "full_with_bg": (
            "<|user_query|>\n{task}\n\n"
            "<|background|>\n{context}\n\n"
            "<|assistant|>\n"
        ),
        "full_no_bg": (
            "<|user_query|>\n{task}\n\n"
            "<|assistant|>\n"
        ),
        "cutoff_marker": "<|assistant|>",
    },

    # ── Scientific summarisation / research output style ──
    "research_notes": {
        "full_with_bg": (
            "Directive: {task}\n"
            "Reference Material: {context}\n"
            "Findings:\n"
        ),
        "full_no_bg": (
            "Directive: {task}\n"
            "Findings:\n"
        ),
        "cutoff_marker": "Findings:",
    },

    # ── Code generation and technical explanation format ──
    "code_instruct": {
        "full_with_bg": (
            "-- PROBLEM --\n{task}\n\n"
            "-- SPECIFICATION --\n{context}\n\n"
            "-- SOLUTION --\n"
        ),
        "full_no_bg": (
            "-- PROBLEM --\n{task}\n\n"
            "-- SOLUTION --\n"
        ),
        "cutoff_marker": "-- SOLUTION --",
    },
}


# ---------------------------------------------------------------------------
# PromptComposer
# ---------------------------------------------------------------------------

class PromptComposer:
    """
    Builds fully formatted prompt strings from the FORMAT_CATALOG and
    extracts the generated text from raw model output at decode time.

    Parameters
    ----------
    format_name : str
        Key of the desired format in FORMAT_CATALOG.
    debug_mode : bool
        When True, assembled prompts are emitted at DEBUG log level.
    """

    def __init__(
        self,
        format_name: str = "domain_instruct",
        debug_mode: bool = False,
    ) -> None:
        if format_name not in FORMAT_CATALOG:
            raise ValueError(
                f"Format '{format_name}' not found in FORMAT_CATALOG. "
                f"Available formats: {list(FORMAT_CATALOG.keys())}"
            )
        self._fmt = FORMAT_CATALOG[format_name]
        self._name = format_name
        self._debug = debug_mode
        _log.info("PromptComposer initialised with format '%s'", format_name)

    # ------------------------------------------------------------------
    # Core composition
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        task: str,
        context: Optional[str] = None,
        target_output: Optional[str] = None,
    ) -> str:
        """
        Compose a complete prompt, optionally appending the target answer.

        Parameters
        ----------
        task : str
            The main instruction or question the model must address.
        context : str, optional
            Domain-specific background that supplements the task.
        target_output : str, optional
            Ground-truth answer — appended during training, omitted at
            inference so the model must generate the response itself.

        Returns
        -------
        str
            Fully formatted prompt string ready for tokenisation.
        """
        cleaned_task = task.strip()
        cleaned_ctx = context.strip() if context else None

        if cleaned_ctx:
            body = self._fmt["full_with_bg"].format(
                task=cleaned_task,
                context=cleaned_ctx,
            )
        else:
            body = self._fmt["full_no_bg"].format(task=cleaned_task)

        if target_output:
            body = f"{body}{target_output.strip()}"

        if self._debug:
            _log.debug("Prompt assembled:\n%s", body)

        return body

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    def extract_response(self, raw_decoded: str) -> str:
        """
        Isolate only the model-generated portion from a full decoded string.

        The model reproduces the entire prompt prefix before its answer.
        We split on the cutoff_marker and return the last segment.

        Parameters
        ----------
        raw_decoded : str
            The full string decoded from model output token IDs.

        Returns
        -------
        str
            Whitespace-stripped generated response, without the prompt.
        """
        marker = self._fmt["cutoff_marker"]
        parts = raw_decoded.split(marker)

        if len(parts) < 2:
            _log.warning(
                "Cutoff marker '%s' not found in decoded output — "
                "returning full string as fallback.",
                marker,
            )
            return raw_decoded.strip()

        return parts[-1].strip()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_format(self) -> str:
        """Name of the currently loaded prompt format."""
        return self._name

    @staticmethod
    def list_formats() -> list[str]:
        """Return all registered format names in FORMAT_CATALOG."""
        return list(FORMAT_CATALOG.keys())
