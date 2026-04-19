#!/usr/bin/env python3
"""
inference_test.py — Test the fine-tuned IPL Analytics Assistant
================================================================
Run from project root:
    python -m finetuning.scripts.inference_test
    python -m finetuning.scripts.inference_test --adapter finetuning/outputs/final_adapter
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

TEST_PROMPTS = [
    {"instruction": "What are V Kohli's IPL career stats?",
     "input": "", "type": "factual"},
    {"instruction": "Why are death overs the highest-scoring phase in IPL?",
     "input": "", "type": "reasoning"},
    {"instruction": "What does strike rate mean for a T20 batter and why does it matter?",
     "input": "", "type": "explanation"},
    {"instruction": "Compare V Kohli and RG Sharma in the IPL.",
     "input": "", "type": "comparison"},
    {"instruction": "Answer the follow-up question based on the conversation context.",
     "input": "Context: User asked about top IPL run scorers.\nFollow-up: Tell me more about MS Dhoni.",
     "type": "followup"},
    {"instruction": "Answer the cricket analytics question using the provided context.",
     "input": ("Question: Should a captain bat or field first at M Chinnaswamy Stadium?\n"
               "Context: M Chinnaswamy Stadium — 65 matches, bat-first win% 41.9%, "
               "avg 1st innings 168, 2nd innings 146, boundary% 18.1%."),
     "type": "rag"},
]


def load_finetuned(adapter_path: str):
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    load_dtype    = getattr(torch, MODEL_LOAD_DTYPE)
    bnb = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=load_dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, instruction, input_text="", max_new=300, temp=0.3):
    prompt = format_prompt_inference(instruction, input_text)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            temperature=temp, do_sample=temp > 0, top_p=0.9,
            repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(OUTPUT_DIR / "final_adapter"))
    args = ap.parse_args()

    model, tok = load_finetuned(args.adapter)
    print("\n" + "=" * 70)
    print("IPL ANALYTICS — INFERENCE TEST")
    print("=" * 70)
    for p in TEST_PROMPTS:
        print(f"\n{'─'*60}\n[{p['type'].upper()}] {p['instruction'][:80]}")
        if p["input"]:
            print(f"  Input: {p['input'][:100]}…")
        print(f"{'─'*60}")
        print(generate(model, tok, p["instruction"], p["input"]))
    print("\n✅ Inference test complete")


if __name__ == "__main__":
    main()
