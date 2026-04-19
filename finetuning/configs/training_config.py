"""
IPL Analytics Assistant — Training Configuration
==================================================
All hyperparameters, model settings, LoRA config, and paths.

Base model: mistralai/Mistral-7B-Instruct-v0.3
Default precision: T4-safe (float16 everywhere, bf16 disabled)
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# PATHS  (training_config.py lives at <project>/finetuning/configs/)
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # <project>/
DATA_DIR     = PROJECT_ROOT / "finetuning" / "data"
OUTPUT_DIR   = PROJECT_ROOT / "finetuning" / "outputs"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE   = DATA_DIR / "val.jsonl"
TEST_FILE  = DATA_DIR / "test_eval.jsonl"

# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# --- Quantization (4-bit NF4 via bitsandbytes) ---
# Reduces VRAM from ~14 GB to ~5 GB so the model fits on a T4 (16 GB).
LOAD_IN_4BIT    = True
BNB_4BIT_QUANT_TYPE  = "nf4"
USE_DOUBLE_QUANT     = True          # nested quantization, saves ~0.4 GB

# IMPORTANT — T4-safe precision defaults
# T4 does NOT support bfloat16.  All dtype settings must use float16.
# If you later run on an Ampere+ GPU (A100, L4, RTX 3090+), you can
# change both values below to "bfloat16" / True for slightly better
# numerical stability — but float16 works fine for LoRA training.
BNB_4BIT_COMPUTE_DTYPE = "float16"   # dtype for 4-bit matrix ops
MODEL_LOAD_DTYPE       = "float16"   # dtype when loading model weights
FP16 = True                          # trainer mixed-precision flag
BF16 = False                         # keep False for T4

# ═══════════════════════════════════════════════════════════════════════════
# LoRA
# ═══════════════════════════════════════════════════════════════════════════
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_BIAS           = "none"

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
LEARNING_RATE              = 2e-4
NUM_EPOCHS                 = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4      # reduce to 2 or 1 if OOM
PER_DEVICE_EVAL_BATCH_SIZE  = 4
GRADIENT_ACCUMULATION_STEPS = 4      # effective batch = 4 × 4 = 16
WARMUP_RATIO               = 0.1
WEIGHT_DECAY               = 0.01
MAX_SEQ_LENGTH             = 512     # reduce to 256 if OOM
OPTIM                      = "paged_adamw_8bit"

# Logging / checkpointing
LOGGING_STEPS    = 10
EVAL_STEPS       = 50
SAVE_STEPS       = 50
SAVE_TOTAL_LIMIT = 3

# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE (Mistral [INST]…[/INST])
# ═══════════════════════════════════════════════════════════════════════════

def format_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    """
    Build a Mistral-style prompt.

    Training:   includes output  → model learns to generate it
    Inference:  omit output      → model generates freely
    """
    if input_text.strip():
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {instruction} [/INST]"
    if output:
        return f"{prompt} {output}</s>"
    return prompt


def format_prompt_inference(instruction: str, input_text: str = "") -> str:
    """Format for inference — no output, model will generate."""
    return format_prompt(instruction, input_text, output="")


# ═══════════════════════════════════════════════════════════════════════════
# OOM RECOVERY GUIDE (ordered by impact)
# ═══════════════════════════════════════════════════════════════════════════
# 1. Reduce PER_DEVICE_TRAIN_BATCH_SIZE → 2, then 1
# 2. Reduce MAX_SEQ_LENGTH → 256
# 3. Reduce GRADIENT_ACCUMULATION_STEPS → 2
# 4. Reduce LORA_R → 8
# 5. Remove "k_proj","o_proj" from LORA_TARGET_MODULES
# 6. Save only at end (set SAVE_STEPS very high)
