# InstructTune-LLM

> **Domain-Specific Instruction Tuning of Large Language Models via Low-Rank Adaptation**
> Fine-tune any causal LM on a custom instruction corpus using LoRA — on a single GPU, in hours, not days.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Transformers-4.38%2B-FFD21E)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-00B388)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Problem Statement

General-purpose LLMs are pre-trained on enormous, heterogeneous web corpora. This makes them broadly capable but poorly calibrated for specialised domains — medical question answering, legal clause extraction, technical documentation generation, financial analysis, or code synthesis in niche frameworks.

Adapting these models to a specific domain introduces a hard engineering trade-off:

| Adaptation Strategy | Core Limitation |
|---|---|
| **Full fine-tuning** | Requires 80–160 GB VRAM, days of A100 compute, and risks catastrophic forgetting of general capabilities |
| **Prompt engineering / few-shot** | Bounded by the context window, brittle on diverse queries, zero lasting knowledge injection |
| **RAG (Retrieval-Augmented Generation)** | Requires a retrieval infrastructure; still limited by the base model's instruction-following quality |

**InstructTune-LLM** takes the precision path: **Low-Rank Adaptation (LoRA)** — a parameter-efficient fine-tuning method that surgically injects small trainable matrices into selected attention layers while keeping the entire backbone frozen.

**Why this approach wins:**
- Trains **< 1%** of total parameters (10–50M vs. 7B+)
- Fine-tunable on a **single 16–24 GB consumer GPU**
- Produces **structured, domain-calibrated responses** that generalise across prompt variations
- Adapters are **interchangeable** — load different domain adapters on the same backbone at zero reload cost

---

## Architecture & Approach

### How LoRA Works

For each targeted linear layer with frozen weight matrix **W ∈ ℝ^{d×k}**, LoRA introduces two small matrices:

```
ΔW = A × B      where A ∈ ℝ^{d×r},  B ∈ ℝ^{r×k},  r ≪ min(d, k)

Effective output = W·x + (α/r) · A·B·x
```

Only **A** and **B** are updated during backpropagation. The backbone produces its standard output; the adapter adds a low-rank correction. At inference:
- **Attached mode**: adapter adds its delta at runtime (no architecture change needed)
- **Merged mode**: `merge_and_unload()` bakes ΔW into W permanently — zero latency overhead, no PEFT dependency at serving time

### Prompt Schema

All training examples and inference queries use a consistent three-section format managed by the `PromptComposer` class:

```
### Task:
<the main instruction or question>

### Context:
<optional domain-specific background information>

### Response:
<model output>
```

Four prompt formats are registered out-of-the-box (`domain_instruct`, `chat_style`, `research_notes`, `code_instruct`) and new ones can be added to `FORMAT_CATALOG` in `src/prompt_builder.py` without touching any other file.

### Training Pipeline

```
Raw JSON Dataset
      │
      ▼
RecordIngester              ← loads, validates, shuffles
      │
      ▼
PromptComposer              ← formats each record into a prompt string
      │
      ▼
Tokenizer + Loss Masking    ← response-only CE loss (prompt tokens masked to -100)
      │
      ▼
HuggingFace Trainer         ← cosine LR, gradient accumulation, mixed precision
      │
      ▼
LoRA Adapter Output         ← lightweight delta weights saved to outputs/lora_adapter/
```

---

## Project Structure

```
InstructTune-LLM/
│
├── src/
│   ├── __init__.py          # Package marker for module imports
│   ├── train_lora.py        # Fine-tuning entry point (run_finetuning orchestrator)
│   ├── inference.py         # CLI inference engine with REPL + single-shot modes
│   ├── data_loader.py       # RecordIngester: load → validate → tokenise → split
│   └── prompt_builder.py    # PromptComposer: FORMAT_CATALOG + build/extract logic
│
├── configs/
│   └── training_config.yaml # Single source of truth for ALL hyperparameters
│
├── data/
│   └── instruction_dataset.json  # 10 high-quality domain instruction examples
│
├── outputs/                 # Auto-created at runtime
│   ├── checkpoints/         # Intermediate Trainer checkpoints
│   ├── lora_adapter/        # Final adapter weights (adapter_model.bin + config)
│   └── logs/                # TensorBoard training logs
│
├── requirements.txt
├── setup.py
└── README.md
```

### Module Responsibilities

| Module | Class / Function | Responsibility |
|---|---|---|
| `prompt_builder.py` | `PromptComposer` | Builds training prompts from `FORMAT_CATALOG`; strips prompt prefix from model output |
| `data_loader.py` | `RecordIngester` | Loads JSON, tokenises records, applies response-only loss masking, creates train/val splits |
| `train_lora.py` | `run_finetuning()` | Orchestrates model loading, LoRA injection, dataset prep, and the Trainer loop |
| `inference.py` | `InferenceSession` | Loads backbone + adapter; exposes `generate()` for single/REPL inference |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/InstructTune-LLM.git
cd InstructTune-LLM
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** `bitsandbytes` 8-bit quantisation has limited native Windows support.
> Set `load_8bit: false` in the config, or use WSL2 for full quantisation support.

### 4. Install the package in editable mode

```bash
pip install -e .
```

### 5. Configure your run

Open `configs/training_config.yaml` and set at minimum:

```yaml
model:
  hub_id: "mistralai/Mistral-7B-v0.1"   # ← set your target model
```

### 6. Authenticate with HuggingFace (for gated models like LLaMA 2/3)

```bash
huggingface-cli login
```

---

## Training

### Standard single-GPU run

```bash
python -m src.train_lora --config configs/training_config.yaml
```

The pipeline will:
1. Read all settings from `configs/training_config.yaml`
2. Load the backbone in 8-bit quantised mode (configurable)
3. Inject LoRA adapters into the specified attention projection layers
4. Tokenise the instruction dataset with response-only loss masking
5. Run the HuggingFace Trainer with cosine LR scheduling and gradient accumulation
6. Save only the lightweight adapter weights to `outputs/lora_adapter/`

### Multi-GPU distributed training (torchrun)

```bash
torchrun --nproc_per_node=4 -m src.train_lora --config configs/training_config.yaml
```

Gradient accumulation is automatically adjusted per-GPU to preserve the configured `target_batch_size`.

### Resume from checkpoint

Set in `configs/training_config.yaml`:

```yaml
training:
  resume_from_checkpoint: "./outputs/checkpoints/checkpoint-400"
```

---

## Inference

### Interactive REPL (recommended for exploration)

```bash
python -m src.inference \
    --backbone mistralai/Mistral-7B-v0.1 \
    --adapter  ./outputs/lora_adapter
```

The REPL prompts for a task and optional context, generates a response, and loops until you type `exit`.

### Single-shot generation

```bash
python -m src.inference \
    --backbone    mistralai/Mistral-7B-v0.1 \
    --adapter     ./outputs/lora_adapter \
    --task        "Explain how LoRA reduces GPU memory usage during training." \
    --context     "Audience: ML engineer evaluating PEFT strategies for a 7B model." \
    --max_tokens  300 \
    --temperature 0.2
```

### Full CLI reference

| Flag | Default | Description |
|---|---|---|
| `--backbone` | **required** | HuggingFace model ID or local path |
| `--adapter` | **required** | Directory containing saved LoRA weights |
| `--format` | `domain_instruct` | Prompt format key (must match training config) |
| `--task` | `None` (→ REPL) | Instruction text |
| `--context` | `None` | Optional background context |
| `--max_tokens` | `256` | Maximum new tokens to generate |
| `--temperature` | `0.2` | Sampling temperature (lower = more deterministic) |
| `--top_p` | `0.85` | Nucleus sampling probability |
| `--top_k` | `50` | Top-k vocabulary cutoff |
| `--num_beams` | `1` | Beam search width (1 = ancestral sampling) |
| `--load_8bit` | `False` | 8-bit backbone loading for lower VRAM |

---

## Sample Outputs

### Example 1 — Deep technical explanation

**Task:** What is catastrophic forgetting and how does LoRA mitigate it?
**Context:** Senior data scientist explaining PEFT to a product team.

```
### Response:
Catastrophic forgetting occurs when gradient updates from new data partially
overwrite pretrained representations, degrading performance on prior tasks.
LoRA prevents this by freezing all backbone weights entirely — the original
parameters are never touched. Only the small rank-decomposed matrices A and B
are updated, and their limited capacity provides an implicit regularisation
that prevents the adapter from overfitting the narrow fine-tuning distribution.
The result: domain-specific gains without sacrificing general capability.
```

---

### Example 2 — Code generation

**Task:** Write a Python function that computes BLEU score without external libraries.

```python
### Response:
def compute_bleu(prediction: str, reference: str, max_ngram: int = 4) -> float:
    pred_tokens = prediction.strip().lower().split()
    ref_tokens  = reference.strip().lower().split()
    precision_scores = []
    for n in range(1, max_ngram + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1))
        ref_ngrams  = Counter(tuple(ref_tokens[i:i+n])  for i in range(len(ref_tokens)-n+1))
        clipped = sum(min(c, ref_ngrams[g]) for g, c in pred_ngrams.items())
        total   = max(len(pred_tokens) - n + 1, 0)
        precision_scores.append(clipped / total if total else 0.0)
    bp  = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens)/len(pred_tokens))
    return bp * math.exp(sum(math.log(p) for p in precision_scores) / max_ngram)
```

---

### Example 3 — Hardware and memory guidance

**Task:** Compare QLoRA and LoRA for fine-tuning a 13B model on a single 24 GB GPU.

```
### Response:
QLoRA quantises backbone weights to 4-bit NF4, reducing a 13B model to ~6-7 GB
VRAM versus ~13 GB for LoRA+8bit. On a 24 GB GPU, QLoRA leaves ample headroom
for activations and gradients, making it the correct choice here. The trade-off
is ~20-40% slower training due to dequantisation overhead — an acceptable cost
given that the alternative (LoRA+8bit) may not even fit in memory.
```

---

## Dataset Format

Every training example follows a clean three-field JSON schema:

```json
{
  "task":     "The main instruction or question the model must address.",
  "context":  "Optional domain-specific background that grounds the response.",
  "response": "The target model output used to supervise training."
}
```

The dataset is a JSON array of such records. The 10 curated examples in `data/instruction_dataset.json` cover:

| # | Topic |
|---|---|
| 1 | Gradient checkpointing for memory-efficient fine-tuning |
| 2 | LoRA alpha / rank relationship and tuning guidance |
| 3 | BLEU score implementation from scratch (Python) |
| 4 | Catastrophic forgetting and how LoRA prevents it |
| 5 | Token-by-token streaming simulation |
| 6 | QLoRA vs. LoRA: memory, speed, and quality trade-offs |
| 7 | Cosine vs. linear LR schedule comparison |
| 8 | Transformer attention: training (parallel) vs. inference (sequential + KV cache) |
| 9 | Three strategies to prevent overfitting on small datasets |
| 10 | Merging a LoRA adapter into the backbone for zero-latency deployment |

You can swap in any JSON file following this schema. For large-scale datasets, use a HuggingFace Hub dataset ID in `dataset.file_path`.

---

## Engineering Design Decisions

| Decision | Rationale |
|---|---|
| **YAML-only configuration** | Zero hardcoded hyperparameters. Every experiment is fully reproducible and auditable from a single file. |
| **`PromptComposer` abstraction** | Changing the prompt format requires only a config edit. `FORMAT_CATALOG` acts as a pluggable registry — add any new format in one place. |
| **`RecordIngester` class** | Strict separation between raw data handling and the training loop. The class is independently unit-testable and reusable across different backbone models. |
| **Response-only loss masking** | Setting prompt token labels to -100 focuses the CE loss signal solely on the answer portion, leading to sharper instruction following and avoiding the model learning to reproduce boilerplate prompt text. |
| **4 LoRA target projections** | Adapting q/k/v/o projections (vs. q/v only) gives the adapter higher expressive capacity for domain shift, at a modest parameter increase (4× larger adapter). |
| **`group_by_length: true`** | Bins sequences of similar length together in each mini-batch, dramatically reducing per-step padding waste and increasing effective throughput. |
| **`merge_and_unload()` for deployment** | Production endpoints load the merged backbone without PEFT as a dependency, removing adapter overhead entirely at zero quality cost. |

---

## Roadmap

- [ ] **QLoRA (4-bit NF4)** — add `load_in_4bit` config option via `BitsAndBytesConfig`
- [ ] **DPO alignment stage** — preference-based fine-tuning after supervised instruction tuning
- [ ] **Multi-turn dialogue support** — extend the schema and `FORMAT_CATALOG` for chat-format datasets
- [ ] **Adapter merge CLI** — `python -m src.merge_adapter --adapter outputs/lora_adapter --output outputs/merged`
- [ ] **Automatic evaluation** — integrate `lm-evaluation-harness` for zero-shot benchmark reporting
- [ ] **HuggingFace Hub upload** — push trained adapters to the Hub after training with a single config flag
- [ ] **Streamlit demo UI** — interactive web interface for stakeholder demonstrations

---

## Requirements

```
torch>=2.1.0
transformers>=4.38.0
datasets>=2.18.0
accelerate>=0.28.0
peft>=0.9.0
bitsandbytes>=0.43.0
pyyaml>=6.0
sentencepiece>=0.2.0
protobuf>=4.25.0
wandb>=0.16.0        # optional — only needed when wandb.enabled: true
```

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE) for full terms.

---

## Acknowledgements

Built on the HuggingFace open-source ecosystem:

- [Transformers](https://github.com/huggingface/transformers) — model hub, tokenizers, Trainer
- [PEFT](https://github.com/huggingface/peft) — LoRA, QLoRA adapter implementations
- [Datasets](https://github.com/huggingface/datasets) — efficient data loading and processing
- [Accelerate](https://github.com/huggingface/accelerate) — hardware-agnostic distributed training
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — 8-bit and 4-bit quantisation kernels

**LoRA paper:**
> Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
> *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022
> [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**QLoRA paper:**
> Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
> *"QLoRA: Efficient Finetuning of Quantized LLMs"*, NeurIPS 2023
> [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)