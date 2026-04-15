"""
src/__init__.py
---------------
Package initialisation for InstructTune-LLM.

Exposes the three public-facing classes at the package level so that
callers can import them directly from `src` rather than navigating
into sub-modules.

Example
-------
    from src import PromptComposer, RecordIngester, InferenceSession
"""

from src.prompt_builder import PromptComposer
from src.data_loader import RecordIngester
from src.inference import InferenceSession

__all__ = [
    "PromptComposer",
    "RecordIngester",
    "InferenceSession",
]

__version__ = "1.0.0"
__project__ = "InstructTune-LLM"
__description__ = "Domain-Specific Instruction Tuning of LLMs via LoRA"
