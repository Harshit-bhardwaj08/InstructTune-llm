"""
src/train_lora.py
-----------------
Fine-tuning entry point for InstructTune-LLM.

All experiment settings are driven by a YAML configuration file.
Nothing is hardcoded — the script is a pure executor that reads its
instructions from configs/training_config.yaml.

Single-GPU run:
    python -m src.train_lora --config configs/training_config.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 -m src.train_lora \\
             --config configs/training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import transformers
import yaml
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import RecordIngester
from src.prompt_builder import PromptComposer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  [%(levelname)-8s]  %(name)s  —  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
_log = logging.getLogger("instruct_tune.trainer")


# ---------------------------------------------------------------------------
# YAML config utilities
# ---------------------------------------------------------------------------

def read_yaml_config(cfg_path: str) -> dict:
    """
    Parse a YAML configuration file into a nested Python dictionary.

    Raises FileNotFoundError if the path does not exist on disk.
    """
    path_obj = Path(cfg_path)
    if not path_obj.is_file():
        raise FileNotFoundError(
            f"Config file not found at: {cfg_path}\n"
            "Please verify the --config argument."
        )
    with open(path_obj, "r", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    _log.info("Loaded configuration from '%s'", cfg_path)
    return settings


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a human-readable dtype string to a torch.dtype object.

    Supported values: 'float16', 'bfloat16', 'float32'
    """
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. "
            f"Choose from: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str]


# ---------------------------------------------------------------------------
# Model + tokenizer initialisation
# ---------------------------------------------------------------------------

def load_base_model(
    model_id: str,
    use_8bit: bool,
    run_dtype: torch.dtype,
    device_placement: str | dict,
    trust_custom_code: bool = False,
) -> tuple:
    """
    Download (or load from cache) the frozen backbone and its tokenizer.

    The backbone is kept frozen after loading; only LoRA adapter weights
    will be updated during the training loop.

    Parameters
    ----------
    model_id : str
        HuggingFace model hub ID or path to a local model directory.
    use_8bit : bool
        Load the model in 8-bit quantised mode via bitsandbytes.
    run_dtype : torch.dtype
        Compute precision for non-quantised model layers.
    device_placement : str or dict
        Passed directly to `device_map` in from_pretrained.
    trust_custom_code : bool
        Set True for models that ship custom modelling code.
    """
    _log.info("Loading tokenizer for '%s' …", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_custom_code,
    )

    # Some base models are released without a pad token; 0 is a safe default
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # Left-padding ensures the causal attention mask is valid at generation time
    tokenizer.padding_side = "left"

    _log.info(
        "Loading backbone '%s'  (8-bit: %s, dtype: %s) …",
        model_id,
        use_8bit,
        run_dtype,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=use_8bit,
        torch_dtype=run_dtype,
        device_map=device_placement,
        trust_remote_code=trust_custom_code,
    )

    return base_model, tokenizer


# ---------------------------------------------------------------------------
# LoRA adapter injection
# ---------------------------------------------------------------------------

def attach_lora_adapters(
    base_model,
    adapter_rank: int,
    scaling_factor: int,
    adapter_dropout: float,
    target_layers: list[str],
    bias_mode: str,
):
    """
    Freeze the backbone and inject low-rank trainable adapters.

    LoRA decomposes each weight update ΔW into two small matrices A and B
    such that ΔW = A × B, where rank(ΔW) = adapter_rank ≪ min(d, k).
    Only A and B are stored and updated; the backbone remains frozen.

    Parameters
    ----------
    base_model :
        The loaded pretrained model to wrap.
    adapter_rank : int
        Intrinsic rank r of the low-rank decomposition.
    scaling_factor : int
        LoRA alpha — scales the adapter contribution (effective lr = α/r).
    adapter_dropout : float
        Dropout probability applied inside adapter layers.
    target_layers : list[str]
        Names of linear projection modules where adapters are injected.
    bias_mode : str
        Controls which bias parameters are updated: 'none', 'all', or
        'lora_only'.
    """
    _log.info(
        "Attaching LoRA adapters — rank=%d  alpha=%d  dropout=%.3f  "
        "targets=%s",
        adapter_rank, scaling_factor, adapter_dropout, target_layers,
    )

    lora_spec = LoraConfig(
        r=adapter_rank,
        lora_alpha=scaling_factor,
        lora_dropout=adapter_dropout,
        target_modules=target_layers,
        bias=bias_mode,
        task_type=TaskType.CAUSAL_LM,
    )

    # Prepares the backbone for quantised adapter training (gradient checkpointing,
    # dtype casting, and layer norm unfreezing handled internally)
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = get_peft_model(base_model, lora_spec)
    base_model.print_trainable_parameters()

    return base_model


# ---------------------------------------------------------------------------
# Checkpoint restoration
# ---------------------------------------------------------------------------

def reload_adapter_weights(model, ckpt_path: str) -> bool:
    """
    Attempt to restore LoRA adapter weights from a checkpoint directory.

    Looks for adapter_model.bin first, then pytorch_model.bin as fallback.
    Returns True if weights were successfully restored, False otherwise.
    """
    candidates = [
        os.path.join(ckpt_path, "adapter_model.bin"),
        os.path.join(ckpt_path, "pytorch_model.bin"),
    ]
    for weight_file in candidates:
        if os.path.isfile(weight_file):
            _log.info("Restoring adapter weights from '%s' …", weight_file)
            saved = torch.load(weight_file)
            set_peft_model_state_dict(model, saved)
            return True

    _log.warning(
        "No checkpoint weights found in '%s' — initialising adapters fresh.",
        ckpt_path,
    )
    return False


# ---------------------------------------------------------------------------
# Experiment tracking (W&B)
# ---------------------------------------------------------------------------

def configure_experiment_tracking(
    tracking_cfg: dict,
) -> tuple[bool, Optional[str]]:
    """
    Set up Weights & Biases environment variables from the tracking config.

    Returns (tracking_active, run_label_or_None).
    """
    if not tracking_cfg.get("enabled", False):
        return False, None

    project_name = tracking_cfg.get("project", "")
    run_label = tracking_cfg.get("run_label") or None
    watch_mode = tracking_cfg.get("watch_gradients", "")
    upload_model = tracking_cfg.get("upload_model", False)

    if project_name:
        os.environ["WANDB_PROJECT"] = project_name
    if watch_mode:
        os.environ["WANDB_WATCH"] = watch_mode
    os.environ["WANDB_LOG_MODEL"] = "true" if upload_model else "false"

    _log.info(
        "W&B tracking enabled — project='%s'  run='%s'",
        project_name,
        run_label,
    )
    return True, run_label


# ---------------------------------------------------------------------------
# Training summary banner
# ---------------------------------------------------------------------------

def _print_run_summary(settings: dict, effective_grad_accum: int) -> None:
    mdl = settings["model"]
    trn = settings["training"]
    ada = settings["adapter"]
    dat = settings["dataset"]

    _log.info(
        "\n"
        "┌──────────────────────────────────────────────────────┐\n"
        "│     InstructTune-LLM  ▸  Domain LoRA Fine-Tuning     │\n"
        "└──────────────────────────────────────────────────────┘\n"
        "  backbone model      : %s\n"
        "  dataset source      : %s\n"
        "  token length cap    : %d\n"
        "  effective batch     : %d  (per_device=%d × accum=%d)\n"
        "  training epochs     : %d\n"
        "  learning rate       : %s\n"
        "  adapter rank/alpha  : %d / %d   dropout=%.3f\n"
        "  injection targets   : %s\n",
        mdl.get("hub_id"),
        dat.get("file_path"),
        trn["token_length_cap"],
        trn["target_batch_size"],
        trn["per_device_batch"],
        effective_grad_accum,
        trn["total_epochs"],
        trn["peak_lr"],
        ada["rank"],
        ada["alpha"],
        ada["dropout"],
        ada["target_modules"],
    )


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def run_finetuning(config_file: str = "configs/training_config.yaml") -> None:
    """
    Orchestrate the complete LoRA fine-tuning workflow.

    Pipeline
    --------
    1. Parse YAML configuration
    2. Detect distributed / multi-GPU environment
    3. Load backbone and tokenizer
    4. Attach LoRA adapters
    5. Tokenise and split the instruction dataset
    6. Optionally resume from an existing checkpoint
    7. Execute the HuggingFace Trainer loop
    8. Persist the final adapter weights
    """
    settings = read_yaml_config(config_file)

    # -- Section aliases -----------------------------------------------
    mdl_cfg   = settings["model"]
    out_cfg   = settings["output"]
    dat_cfg   = settings["dataset"]
    pmt_cfg   = settings["prompt"]
    ada_cfg   = settings["adapter"]
    trn_cfg   = settings["training"]
    wdb_cfg   = settings.get("wandb", {})

    hub_id: str = mdl_cfg.get("hub_id", "")
    if not hub_id:
        raise ValueError(
            "model.hub_id is required in the config YAML. "
            "Example: 'mistralai/Mistral-7B-v0.1'"
        )

    # -- Distributed training context ----------------------------------
    world_size     = int(os.environ.get("WORLD_SIZE", 1))
    local_rank     = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    is_main_proc   = local_rank == 0

    # Gradient accumulation keeps effective batch size constant
    per_dev_batch  = trn_cfg["per_device_batch"]
    target_bs      = trn_cfg["target_batch_size"]
    grad_accum     = target_bs // per_dev_batch
    if is_distributed:
        grad_accum = max(1, grad_accum // world_size)

    placement = "auto" if not is_distributed else {"": local_rank}

    # -- Experiment tracking -------------------------------------------
    use_tracking, run_label = configure_experiment_tracking(wdb_cfg)

    # -- Run summary ---------------------------------------------------
    if is_main_proc:
        _print_run_summary(settings, grad_accum)

    # -- Backbone + tokenizer ------------------------------------------
    compute_dtype = resolve_torch_dtype(
        mdl_cfg.get("precision_dtype", "float16")
    )
    backbone, tokenizer = load_base_model(
        model_id=hub_id,
        use_8bit=mdl_cfg.get("load_8bit", True),
        run_dtype=compute_dtype,
        device_placement=placement,
        trust_custom_code=mdl_cfg.get("trust_remote_code", False),
    )

    # -- Attach LoRA adapters ------------------------------------------
    backbone = attach_lora_adapters(
        base_model=backbone,
        adapter_rank=ada_cfg["rank"],
        scaling_factor=ada_cfg["alpha"],
        adapter_dropout=ada_cfg["dropout"],
        target_layers=ada_cfg["target_modules"],
        bias_mode=ada_cfg.get("bias_update", "none"),
    )

    # -- Dataset preparation -------------------------------------------
    composer = PromptComposer(
        format_name=pmt_cfg.get("format_name", "domain_instruct")
    )
    ingester = RecordIngester(
        file_path=dat_cfg["file_path"],
        tokenizer=tokenizer,
        composer=composer,
        column_map=dat_cfg.get("column_map"),
        max_seq_len=trn_cfg["token_length_cap"],
        val_set_size=dat_cfg.get("val_set_size", 200),
        response_only_loss=trn_cfg.get("response_only_loss", True),
        append_eos_token=trn_cfg.get("append_eos", True),
        shuffle_seed=dat_cfg.get("seed", 42),
    )
    train_split, val_split = ingester.prepare_dataset()

    # -- Optional checkpoint resume ------------------------------------
    resume_path = trn_cfg.get("resume_from_checkpoint")
    if resume_path:
        loaded_ok = reload_adapter_weights(backbone, resume_path)
        if not loaded_ok:
            resume_path = None  # fall back to fresh start

    # -- Multi-GPU model parallelism (non-DDP) -------------------------
    if not is_distributed and torch.cuda.device_count() > 1:
        backbone.is_parallelizable = True
        backbone.model_parallel    = True

    # -- Output directories --------------------------------------------
    ckpt_dir    = out_cfg["checkpoint_dir"]
    weights_dir = out_cfg["adapter_dir"]
    log_dir     = out_cfg["log_dir"]
    for directory in (ckpt_dir, weights_dir, log_dir):
        os.makedirs(directory, exist_ok=True)

    # -- TrainingArguments ---------------------------------------------
    has_val         = val_split is not None
    eval_interval   = trn_cfg.get("eval_every_n_steps", 200)
    save_interval   = trn_cfg.get("save_every_n_steps", 200)

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=per_dev_batch,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=trn_cfg.get("warmup_steps", 100),
        num_train_epochs=trn_cfg["total_epochs"],
        learning_rate=trn_cfg["peak_lr"],
        lr_scheduler_type=trn_cfg.get("lr_schedule", "cosine"),
        fp16=(trn_cfg.get("mixed_precision", "fp16") == "fp16"),
        bf16=(trn_cfg.get("mixed_precision", "fp16") == "bf16"),
        logging_steps=10,
        logging_dir=log_dir,
        optim=trn_cfg.get("optimiser", "adamw_torch"),
        evaluation_strategy="steps" if has_val else "no",
        save_strategy="steps",
        eval_steps=eval_interval if has_val else None,
        save_steps=save_interval,
        output_dir=ckpt_dir,
        save_total_limit=trn_cfg.get("keep_n_checkpoints", 3),
        load_best_model_at_end=has_val,
        ddp_find_unused_parameters=False if is_distributed else None,
        group_by_length=trn_cfg.get("group_by_length", False),
        report_to="wandb" if use_tracking else "none",
        run_name=run_label,
    )

    # -- Trainer build -------------------------------------------------
    # Disable KV cache — incompatible with gradient checkpointing
    backbone.config.use_cache = False

    # Override state_dict to serialise only the LoRA delta, not the full model
    _orig_state_dict = backbone.state_dict
    backbone.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, _orig_state_dict()
        )
    ).__get__(backbone, type(backbone))

    # torch.compile gives ~15% throughput uplift on PyTorch >= 2.0 (Linux/Mac)
    if torch.__version__ >= "2" and sys.platform != "win32":
        _log.info("Applying torch.compile() for graph-level kernel fusion …")
        backbone = torch.compile(backbone)

    trainer = transformers.Trainer(
        model=backbone,
        train_dataset=train_split,
        eval_dataset=val_split,
        args=train_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,  # align tensors for efficient CUDA kernels
            return_tensors="pt",
            padding=True,
        ),
    )

    _log.info("Starting fine-tuning run …")
    trainer.train(resume_from_checkpoint=resume_path)

    # -- Persist adapter weights ---------------------------------------
    backbone.save_pretrained(weights_dir)
    _log.info("LoRA adapter saved to '%s'", weights_dir)
    _log.info("Fine-tuning complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="InstructTune-LLM — domain-specific LoRA fine-tuning runner"
    )
    cli.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the YAML configuration file (default: configs/training_config.yaml).",
    )
    parsed = cli.parse_args()
    run_finetuning(config_file=parsed.config)
