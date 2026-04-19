#!/usr/bin/env python3
"""
compare_base_vs_lora.py — Side-by-side base vs fine-tuned comparison
=====================================================================
Run from project root:
    python -m finetuning.scripts.compare_base_vs_lora
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from finetuning.configs.training_config import (
    BASE_MODEL, OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE,
    USE_DOUBLE_QUANT, MODEL_LOAD_DTYPE, format_prompt_inference,
)

PROMPTS = [
    {"instruction": "Who are the all-time top IPL run scorers?",
     "input": "", "type": "factual"},
    {"instruction": "Why does the Powerplay have the highest dot ball percentage?",
     "input": "", "type": "reasoning"},
    {"instruction": "What does boundary percentage indicate?",
     "input": "", "type": "explanation"},
    {"instruction": "Compare JJ Bumrah and YS Chahal as IPL bowlers.",
     "input": "", "type": "comparison"},
    {"instruction": "Answer the cricket analytics question using the provided context.",
     "input": "Question: Is SP Narine economical?\nContext: SP Narine has an economy of 6.66 and dot ball% of 38.0%.",
     "type": "rag"},
]


def _gen(model, tok, instruction, input_text="", max_new=250):
    prompt = format_prompt_inference(instruction, input_text)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new, temperature=0.3,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(OUTPUT_DIR / "final_adapter"))
    args = ap.parse_args()

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    load_dtype    = getattr(torch, MODEL_LOAD_DTYPE)
    bnb = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT, bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE, bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )

    print(f"📦 Loading base + adapter …")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=load_dtype,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ft = PeftModel.from_pretrained(base, args.adapter)
    ft.eval()

    print("\n" + "=" * 70)
    print("BASE  vs  FINE-TUNED")
    print("=" * 70)

    for p in PROMPTS:
        print(f"\n{'━'*70}\n[{p['type'].upper()}] {p['instruction'][:80]}")
        if p["input"]:
            print(f"  Input: {p['input'][:100]}…")
        print(f"{'━'*70}")

        with ft.disable_adapter():
            base_resp = _gen(ft, tok, p["instruction"], p["input"])
        ft_resp = _gen(ft, tok, p["instruction"], p["input"])

        print(f"\n🔵 BASE:\n  {base_resp[:300]}")
        print(f"\n🟢 FINE-TUNED:\n  {ft_resp[:300]}")

    print("\n✅ Comparison complete")


if __name__ == "__main__":
    main()
