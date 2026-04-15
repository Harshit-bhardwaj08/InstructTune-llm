"""
src/inference.py
----------------
CLI-based inference engine for InstructTune-LLM.

Loads a frozen backbone alongside the fine-tuned LoRA adapter weights and
supports two operating modes:

    Interactive REPL — prompts the user for each task & context pair and
        streams the response to stdout.  Good for rapid exploration.

    Single-shot — accepts task/context as CLI flags and exits after one
        generation.  Suitable for scripting and pipeline integration.

Usage
-----
Interactive:
    python -m src.inference \\
        --backbone mistralai/Mistral-7B-v0.1 \\
        --adapter  ./outputs/lora_adapter

Single-shot:
    python -m src.inference \\
        --backbone mistralai/Mistral-7B-v0.1 \\
        --adapter  ./outputs/lora_adapter \\
        --task     "Summarise the LoRA paper in three sentences." \\
        --context  "Audience: undergraduate ML student."
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.prompt_builder import PromptComposer

logging.basicConfig(
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
_log = logging.getLogger("instruct_tune.inference")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_compute_device() -> str:
    """Return the fastest available compute device identifier."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass  # MPS not available on this PyTorch build
    return "cpu"


# ---------------------------------------------------------------------------
# Inference session
# ---------------------------------------------------------------------------

class InferenceSession:
    """
    Encapsulates a backbone + LoRA adapter pair and exposes a clean
    `generate()` method for prompt-conditioned text generation.

    Parameters
    ----------
    model_hub_id : str
        HuggingFace model ID or local path of the base (non-finetuned) model.
    adapter_path : str
        Directory containing the saved LoRA adapter weights.
    load_8bit : bool
        Load the backbone in 8-bit quantised mode (lower VRAM usage).
    prompt_format : str
        Prompt format key matching what was used during training.
    """

    def __init__(
        self,
        model_hub_id: str,
        adapter_path: str,
        load_8bit: bool = False,
        prompt_format: str = "domain_instruct",
    ) -> None:
        self._device = select_compute_device()
        _log.info("Compute device selected: %s", self._device.upper())

        self._composer = PromptComposer(format_name=prompt_format)
        self._tokenizer, self._model = self._initialise_model(
            model_hub_id, adapter_path, load_8bit
        )

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _initialise_model(
        self,
        model_hub_id: str,
        adapter_path: str,
        load_8bit: bool,
    ):
        """Load the tokenizer, frozen backbone, and adapter weights."""
        _log.info("Loading tokenizer: '%s' …", model_hub_id)
        tokenizer = AutoTokenizer.from_pretrained(model_hub_id)

        # Standardise special token IDs for consistent behaviour
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

        # Use fp16 on GPU/MPS for speed; fp32 on CPU for numerical stability
        weight_dtype = (
            torch.float16 if self._device in ("cuda", "mps") else torch.float32
        )
        device_map = "auto" if self._device == "cuda" else {"": self._device}

        _log.info("Loading base model weights …")
        frozen_base = AutoModelForCausalLM.from_pretrained(
            model_hub_id,
            load_in_8bit=load_8bit,
            torch_dtype=weight_dtype,
            device_map=device_map,
        )

        _log.info("Attaching LoRA adapter from '%s' …", adapter_path)
        adapted_model = PeftModel.from_pretrained(
            frozen_base,
            adapter_path,
            torch_dtype=weight_dtype,
        )

        # Keep special token IDs consistent in the model config
        adapted_model.config.pad_token_id = tokenizer.pad_token_id
        adapted_model.config.bos_token_id = tokenizer.bos_token_id
        adapted_model.config.eos_token_id = tokenizer.eos_token_id

        # Convert to fp16 unless running in 8-bit mode
        if not load_8bit:
            adapted_model = adapted_model.half()

        adapted_model.eval()

        # Graph-level optimisation on PyTorch >= 2 (Linux/Mac only)
        if torch.__version__ >= "2" and sys.platform != "win32":
            _log.info("torch.compile() applied for kernel-level inference speedup.")
            adapted_model = torch.compile(adapted_model)

        _log.info("Model loaded and ready.")
        return tokenizer, adapted_model

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        task: str,
        context: str | None = None,
        *,
        temperature: float = 0.2,
        top_p: float = 0.85,
        top_k: int = 50,
        num_beams: int = 1,
        max_new_tokens: int = 256,
        rep_penalty: float = 1.1,
    ) -> str:
        """
        Generate a structured response for a given task / context pair.

        Parameters
        ----------
        task : str
            Main query or instruction the model should respond to.
        context : str, optional
            Background information to ground the response.
        temperature : float
            Sampling temperature — lower values are more deterministic.
        top_p : float
            Nucleus sampling cumulative probability threshold.
        top_k : int
            Restrict sampling to the top-k highest probability tokens.
        num_beams : int
            Beam search width; 1 uses ancestral sampling.
        max_new_tokens : int
            Hard cap on the number of tokens the model may generate.
        rep_penalty : float
            Penalise repeated tokens to reduce looping/repetition.

        Returns
        -------
        str
            The generated response text, with the prompt prefix stripped.
        """
        full_prompt = self._composer.build_prompt(task=task, context=context)

        input_encoding = self._tokenizer(full_prompt, return_tensors="pt")
        token_ids = input_encoding["input_ids"].to(self._device)

        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=rep_penalty,
            do_sample=(num_beams == 1),
        )

        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=token_ids,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
            )

        raw_decoded = self._tokenizer.decode(
            generation_output.sequences[0],
            skip_special_tokens=True,
        )
        return self._composer.extract_response(raw_decoded)


# ---------------------------------------------------------------------------
# CLI argument definition
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instruct-tune-infer",
        description=(
            "InstructTune-LLM  ▸  Domain LoRA Inference CLI\n"
            "Generate structured responses using a fine-tuned causal LM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        required=True,
        help="HuggingFace model ID or local path to the base model.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the directory containing saved LoRA adapter weights.",
    )
    parser.add_argument(
        "--format",
        default="domain_instruct",
        choices=PromptComposer.list_formats(),
        help="Prompt format key — must match the one used during training.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Instruction text.  Omit to launch interactive REPL mode.",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional background context sent alongside the task.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.85,
        help="Nucleus sampling probability (default: 0.85).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k token sampling cutoff (default: 50).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search width; 1 enables sampling mode (default: 1).",
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load backbone in 8-bit quantised mode for lower VRAM usage.",
    )
    return parser


# ---------------------------------------------------------------------------
# Response display
# ---------------------------------------------------------------------------

def render_response(response_text: str) -> None:
    """Pretty-print the generated response with terminal decorators."""
    bar = "─" * 74
    print(f"\n{bar}")
    print("  ▸  MODEL RESPONSE")
    print(bar)
    print(response_text)
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def launch_repl(session: InferenceSession, args: argparse.Namespace) -> None:
    """
    Run an interactive prompt loop for exploratory inference.
    Type 'exit', 'quit', or press Ctrl-C to terminate the session.
    """
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│   InstructTune-LLM  ▸  Interactive Inference Mode   │")
    print("│   Enter 'exit' or 'quit' at any time to stop.       │")
    print("└─────────────────────────────────────────────────────┘\n")

    while True:
        try:
            task_input = input("### Task (required):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if task_input.lower() in ("exit", "quit", "q", ":q"):
            print("Goodbye.")
            break
        if not task_input:
            print("[!] Task field cannot be empty — please try again.\n")
            continue

        context_input = input(
            "### Context (optional — press Enter to skip):\n> "
        ).strip()
        background = context_input if context_input else None

        print("\n[Generating response …]")
        output = session.generate(
            task=task_input,
            context=background,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=args.max_tokens,
        )
        render_response(output)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    session = InferenceSession(
        model_hub_id=args.backbone,
        adapter_path=args.adapter,
        load_8bit=args.load_8bit,
        prompt_format=args.format,
    )

    if args.task:
        _log.info("Single-shot generation mode.")
        output = session.generate(
            task=args.task,
            context=args.context,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=args.max_tokens,
        )
        render_response(output)
    else:
        launch_repl(session, args)


if __name__ == "__main__":
    main()
