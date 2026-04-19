#!/usr/bin/env python3
"""
evaluate_model.py — Evaluation on test set
============================================
Run from project root:
    python -m finetuning.scripts.evaluate_model
    python -m finetuning.scripts.evaluate_model --max-samples 10   # quick check
"""

import argparse, json, re
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from finetuning.configs.training_config import (
    BASE_MODEL, OUTPUT_DIR, TEST_FILE,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE,
    USE_DOUBLE_QUANT, MODEL_LOAD_DTYPE, format_prompt_inference,
)
from finetuning.scripts.dataset_loader import load_jsonl


def _load(adapter):
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    load_dtype    = getattr(torch, MODEL_LOAD_DTYPE)
    bnb = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT, bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE, bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=load_dtype,
    )
    m = PeftModel.from_pretrained(m, adapter); m.eval()
    t = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return m, t


def _gen(m, t, instr, inp="", max_new=300):
    prompt = format_prompt_inference(instr, inp)
    ids = t(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(
            **ids, max_new_tokens=max_new, temperature=0.1,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=t.pad_token_id,
        )
    return t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _metrics(expected, predicted):
    lr = len(predicted) / max(len(expected), 1)
    ew = set(expected.lower().split())
    pw = set(predicted.lower().split())
    wr = len(ew & pw) / len(ew) if ew else 0
    en = set(re.findall(r'\d+\.?\d*', expected))
    pn = set(re.findall(r'\d+\.?\d*', predicted))
    nr = len(en & pn) / len(en) if en else 1.0
    return {"length_ratio": round(lr, 2), "word_recall": round(wr, 3), "number_recall": round(nr, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(OUTPUT_DIR / "final_adapter"))
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    m, t = _load(args.adapter)
    test_raw = load_jsonl(TEST_FILE)
    if args.max_samples:
        test_raw = test_raw[:args.max_samples]
    print(f"\n📊 Evaluating {len(test_raw)} samples …")

    results = []
    for i, s in enumerate(test_raw):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_raw)} …")
        pred = _gen(m, t, s["instruction"], s.get("input", ""))
        met = _metrics(s["output"], pred)
        results.append({
            "index": i, "type": s["meta"]["type"], "topic": s["meta"]["topic"],
            "instruction": s["instruction"], "input": s.get("input", ""),
            "expected": s["output"], "predicted": pred, "metrics": met,
        })

    out_path = OUTPUT_DIR / "evaluation_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Aggregate
    type_m = {}
    for r in results:
        tp = r["type"]
        type_m.setdefault(tp, {"wr": [], "nr": [], "n": 0})
        type_m[tp]["wr"].append(r["metrics"]["word_recall"])
        type_m[tp]["nr"].append(r["metrics"]["number_recall"])
        type_m[tp]["n"] += 1

    lines = ["=" * 60, "EVALUATION SUMMARY", "=" * 60, f"Samples: {len(results)}", ""]
    all_wr, all_nr = [], []
    for tp in sorted(type_m):
        d = type_m[tp]
        awr = sum(d["wr"]) / len(d["wr"]); anr = sum(d["nr"]) / len(d["nr"])
        lines.append(f"  {tp:15s}  n={d['n']:3d}  word_recall={awr:.3f}  number_recall={anr:.3f}")
        all_wr += d["wr"]; all_nr += d["nr"]
    lines += ["", f"Overall word_recall={sum(all_wr)/len(all_wr):.3f}  number_recall={sum(all_nr)/len(all_nr):.3f}",
              "", "NOTE: Full quality evaluation requires manual review.", "=" * 60]
    summary = "\n".join(lines)
    print(f"\n{summary}")

    (OUTPUT_DIR / "evaluation_summary.txt").write_text(summary)
    print(f"💾 Saved to: {out_path}")


if __name__ == "__main__":
    main()
