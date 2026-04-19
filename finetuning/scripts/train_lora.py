#!/usr/bin/env python3
"""
train_lora.py — LoRA Fine-Tuning for IPL Analytics Assistant
==============================================================
Uses standard HuggingFace Trainer (not SFTTrainer) because the
dataset is already tokenized with label masking in dataset_loader.py.

Run from project root:
    python -m finetuning.scripts.train_lora                  # full training
    python -m finetuning.scripts.train_lora --smoke-test     # quick validation
    python -m finetuning.scripts.train_lora --resume         # continue from checkpoint
    python -m finetuning.scripts.train_lora --checkpoint finetuning/outputs/lora_run/checkpoint-50
"""

import argparse, sys, os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from finetuning.configs.training_config import (
    BASE_MODEL, OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE,
    USE_DOUBLE_QUANT, MODEL_LOAD_DTYPE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LORA_BIAS,
    LEARNING_RATE, NUM_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    WARMUP_RATIO, WEIGHT_DECAY, MAX_SEQ_LENGTH,
    LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT,
    FP16, BF16, OPTIM,
    TRAIN_FILE, VAL_FILE,
)
from finetuning.scripts.dataset_loader import load_and_prepare_datasets


# ═══════════════════════════════════════════════════════════════════════════
# PREFLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def preflight(tokenizer, model):
    """Validate everything before training starts. Exit early on failure."""
    errors = []

    # 1. Dataset files exist and are non-empty
    for name, path in [("train", TRAIN_FILE), ("val", VAL_FILE)]:
        if not path.exists():
            errors.append(f"{name} file not found: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{name} file is empty: {path}")

    # 2. Output directory is writable
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"Cannot create output dir {OUTPUT_DIR}: {e}")

    # 3. Tokenizer can encode a sample
    try:
        test_text = "[INST] Test prompt [/INST] Test response</s>"
        ids = tokenizer(test_text, truncation=True, max_length=MAX_SEQ_LENGTH)
        assert len(ids["input_ids"]) > 0, "tokenizer produced empty ids"
    except Exception as e:
        errors.append(f"Tokenizer test failed: {e}")

    # 4. LoRA target modules exist in model
    model_modules = {n.split(".")[-1] for n, _ in model.named_modules()}
    for target in LORA_TARGET_MODULES:
        if target not in model_modules:
            errors.append(
                f"LoRA target module '{target}' not found in model. "
                f"Available leaf modules: {sorted(model_modules)[:20]}…"
            )

    # 5. Precision consistency check
    if BF16 and not torch.cuda.is_bf16_supported():
        errors.append(
            "BF16=True but this GPU does not support bfloat16. "
            "Set BF16=False, FP16=True in training_config.py."
        )

    if errors:
        print("\n❌ PREFLIGHT FAILED:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)

    print("  ✓ Preflight checks passed")


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """Load quantized base model, apply LoRA, return (model, tokenizer)."""
    print(f"\n📦 Loading model: {BASE_MODEL}")

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    load_dtype    = getattr(torch, MODEL_LOAD_DTYPE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=load_dtype,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"   Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for IPL Analytics")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in the output dir")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from a specific checkpoint path")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick validation run: 20 train samples, 1 epoch, 2 steps")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples (for debugging)")
    args = parser.parse_args()

    print("=" * 70)
    print("IPL ANALYTICS — LoRA FINE-TUNING" +
          ("  [SMOKE TEST]" if args.smoke_test else ""))
    print("=" * 70)

    # ── Load model & tokenizer ──
    model, tokenizer = load_model_and_tokenizer()

    # ── Preflight ──
    print("\n🔍 Running preflight checks …")
    preflight(tokenizer, model.get_base_model())

    # ── Determine sample limits ──
    max_train = args.max_train_samples
    max_val   = None
    if args.smoke_test:
        max_train = max_train or 20
        max_val   = 10

    # ── Load & prepare data ──
    train_ds, val_ds, _ = load_and_prepare_datasets(
        tokenizer, max_train=max_train, max_val=max_val,
    )
    print(f"\n📊 Dataset ready: {len(train_ds)} train, {len(val_ds)} val")

    # ── Training arguments ──
    run_dir = OUTPUT_DIR / "lora_run"

    num_epochs  = 1 if args.smoke_test else NUM_EPOCHS
    eval_steps  = 2 if args.smoke_test else EVAL_STEPS
    save_steps  = 999_999 if args.smoke_test else SAVE_STEPS   # no checkpoints in smoke
    log_steps   = 1 if args.smoke_test else LOGGING_STEPS
    max_steps   = 4 if args.smoke_test else -1                  # -1 = use num_epochs

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        fp16=FP16,
        bf16=BF16,
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=not args.smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        seed=42,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        pad_to_multiple_of=8,
    )

    # ── Trainer (standard HF Trainer — data is already tokenized + labelled) ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # ── Checkpoint / resume logic ──
    resume_from = None
    if args.checkpoint:
        resume_from = args.checkpoint
        print(f"   Resuming from: {resume_from}")
    elif args.resume:
        checkpoints = sorted(run_dir.glob("checkpoint-*")) if run_dir.exists() else []
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"   Resuming from latest: {resume_from}")
        else:
            print("   No checkpoints found — starting fresh.")

    # ── Train ──
    print("\n🚀 Starting training …")
    trainer.train(resume_from_checkpoint=resume_from)

    if args.smoke_test:
        print("\n✅ Smoke test passed — pipeline is functional.")
        return

    # ── Save final adapter ──
    final_dir = OUTPUT_DIR / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n💾 Final adapter saved to: {final_dir}")

    metrics = trainer.evaluate()
    print(f"📈 Final eval loss: {metrics['eval_loss']:.4f}")

    print("\n✅ Training complete!")
    print(f"   Adapter: {final_dir}")
    print(f"   Test:  python -m finetuning.scripts.inference_test")


if __name__ == "__main__":
    main()
