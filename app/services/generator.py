"""
Generator v3.1 — Answer quality polish
========================================
Changes from v3 (ONLY these two areas):

1. INSIGHT / GENERAL prompt: rewritten to produce natural human-style
   explanations instead of formula definitions. Structure:
     - Plain English explanation (no formulas)
     - Why it matters in T20/IPL
     - Then one brief example from data

2. MATCHUP prompt + verdict builder: enriched to produce interpreted
   answers instead of raw data dumps. Structure:
     - Head-to-head stats with plain English interpretation
     - What the SR and dismissals actually mean in context
     - Clear final verdict (batter edge vs bowler edge)

Everything else is UNCHANGED from v3:
  - Model loading, _safe_float, _parse_player_line
  - _build_comparison_verdict (comparison logic is correct, not touched)
  - All other intent prompts (leaderboard, player_lookup, comparison,
    season_comparison, venue, phase, season)
  - _build_prompt logic for comparison injection
  - Generation parameters (temp=0.2, top_p=0.85, rep_penalty=1.15)
"""

import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_PROJECT = Path(__file__).resolve().parent.parent.parent
_ADAPTER_DIR = _PROJECT / "finetuning" / "outputs" / "final_adapter"

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

_MODEL = None
_TOKENIZER = None
_IS_FINETUNED = False
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ═══════════════════════════════════════════════════════════════════════════
# Model loading (UNCHANGED from v3)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(adapter_path: str = None, force_base: bool = False):
    global _MODEL, _TOKENIZER, _IS_FINETUNED
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    print(f"📦 Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    adapter = adapter_path or str(_ADAPTER_DIR)
    if not force_base and _HAS_PEFT and Path(adapter).exists():
        print(f"📎 Loading LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        _IS_FINETUNED = True
    else:
        print(f"⚠ Running with BASE model only")
        _IS_FINETUNED = False
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        adapter if _IS_FINETUNED and Path(adapter).exists() else BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _MODEL, _TOKENIZER = model, tokenizer
    return model, tokenizer


def is_finetuned() -> bool:
    return _IS_FINETUNED


# ═══════════════════════════════════════════════════════════════════════════
# Context parsing (UNCHANGED from v3)
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(s: str) -> float | None:
    s = s.strip().rstrip(".,;:!?")
    if s == "N/A" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_player_line(line: str) -> dict:
    result = {}
    if line.startswith("Player A:") or line.startswith("Player B:"):
        rest = line.split(":", 1)[1].strip()
        colon2 = rest.find(":")
        if colon2 > 0:
            result["name"] = rest[:colon2].strip()
    else:
        colon = line.find(":")
        if colon > 0:
            result["name"] = line[:colon].strip()

    patterns = [
        ("runs",         r"(\d+)\s+runs"),
        ("wickets",      r"(\d+)\s+wickets"),
        ("matches",      r"(\d+)\s+matches"),
        ("balls",        r"(\d+)\s+balls"),
        ("sr",           r"SR\s+([\d.]+)"),
        ("economy",      r"Economy\s+([\d.]+)"),
        ("avg",          r"avg\s+([\d.]+|N/A)"),
        ("boundary_pct", r"[Bb]oundary%\s+([\d.]+)"),
        ("dot_pct",      r"[Dd]ot%?\s+([\d.]+)"),
        ("sixes",        r"(\d+)\s+sixes"),
        ("fours",        r"(\d+)\s+fours"),
        ("dots",         r"(\d+)\s+dots"),
        ("dismissed",    r"[Dd]ismissed\s+(\d+)"),
        ("highest",      r"[Hh]ighest:?\s+(\d+)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, line)
        if m:
            result[key] = _safe_float(m.group(1))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pre-computed comparison verdict (UNCHANGED from v3)
# ═══════════════════════════════════════════════════════════════════════════

def _build_comparison_verdict(context: str) -> str:
    lines = context.strip().split("\n")
    players = []
    for line in lines:
        if line.startswith("Player A:") or line.startswith("Player B:"):
            parsed = _parse_player_line(line)
            if parsed.get("name"):
                players.append(parsed)

    if len(players) < 2:
        return ""

    a, b = players[0], players[1]
    na = a.get("name", "Player A")
    nb = b.get("name", "Player B")

    is_bowling = ("wickets" in a and "economy" in a
                  and "runs" not in a)

    if is_bowling:
        metrics = [
            ("wickets",  "Wickets",      True),
            ("economy",  "Economy",      False),
            ("avg",      "Bowling avg",  False),
            ("sr",       "Bowling SR",   False),
            ("dot_pct",  "Dot ball%",    True),
        ]
    else:
        metrics = [
            ("runs",         "Runs",        True),
            ("sr",           "Strike rate", True),
            ("avg",          "Batting avg", True),
            ("boundary_pct", "Boundary%",   True),
            ("dot_pct",      "Dot ball%",   False),
            ("sixes",        "Sixes",       True),
        ]

    verdicts = []
    for key, label, higher_is_better in metrics:
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            continue
        if abs(va - vb) < 0.001:
            verdicts.append(f"• {label}: tied at {va}")
        elif (va > vb) == higher_is_better:
            verdicts.append(f"• {label}: {na} leads ({va} vs {vb})")
        else:
            verdicts.append(f"• {label}: {nb} leads ({vb} vs {va})")

    if not verdicts:
        return ""

    header = (
        f"Pre-computed comparison (TRUST THESE — do not invert):\n"
        f"Type: {'BOWLING' if is_bowling else 'BATTING'} comparison\n"
    )
    return header + "\n".join(verdicts)


# ═══════════════════════════════════════════════════════════════════════════
# Pre-computed matchup verdict (IMPROVED — v3.1)
# ═══════════════════════════════════════════════════════════════════════════
# Change: verdict now includes human-readable interpretation sentences
# that the LLM can weave into its answer rather than just raw labels.

def _build_matchup_verdict(context: str) -> str:
    """
    Extract interpretation hints from the enriched matchup context.
    The context_builder already computed all derived metrics and verdicts.
    We just extract and relay the key guidance lines.
    """
    # The context now has three sections. We look for "Interpretation Hints:"
    # and "Derived Matchup Metrics:" which are pre-computed deterministically.
    lines = context.strip().split("\n")

    # Find batter/bowler names from raw data line
    batter = "the batter"
    bowler = "the bowler"
    for line in lines:
        if " vs " in line and "balls" in line and ":" in line:
            parts = line.split(" vs ")
            if len(parts) >= 2:
                batter = parts[0].strip().split(":")[-1].strip()
                if not batter:
                    batter = parts[0].strip()
                bowler = parts[1].split(":")[0].strip()
            break

    guide = [
        f"BATTER is {batter}. BOWLER is {bowler}.",
        "",
        "The context below contains pre-computed Derived Matchup Metrics and",
        "Interpretation Hints. USE THEM in your answer — do not recalculate.",
        "Every number you cite should come from the Derived Matchup Metrics section.",
        "",
        "RULES:",
        "• Do NOT compare batter's career runs with bowler's career wickets.",
        "• Do NOT invent numbers — use only what is in the context.",
    ]

    return "\n".join(guide)


# ═══════════════════════════════════════════════════════════════════════════
# Intent-specific prompt templates
# ═══════════════════════════════════════════════════════════════════════════
# CHANGED: "general" and "matchup" templates rewritten for quality.
# ALL OTHER templates are IDENTICAL to v3.

_INTENT_INSTRUCTIONS = {

    # ── UNCHANGED from v3 ──

    "leaderboard": (
        "You are an IPL analytics assistant. Use ONLY the provided data.\n"
        "List each player with their rank and key stats.\n"
        "End with one sentence of interpretation.\n"
        "Do NOT invent numbers. Do NOT add players not in the data."
    ),

    "player_lookup": (
        "You are an IPL analytics assistant. Use ONLY the provided data.\n"
        "Answer with:\n"
        "• Key stats (runs/wickets, SR/economy, average, dot%)\n"
        "• One notable strength visible in the numbers\n"
        "• One brief analytical insight\n"
        "Do NOT invent or estimate any number not in the data."
    ),

    "comparison": (
        "You are an IPL analytics assistant. Compare using ONLY the provided data.\n"
        "\n"
        "STRICT COMPARISON RULES:\n"
        "1. A pre-computed comparison is provided below — TRUST IT EXACTLY.\n"
        "2. Copy the metric-by-metric verdicts into your answer as bullet points.\n"
        "3. NEVER say a smaller number is 'higher' or 'more' — that is WRONG.\n"
        "4. NEVER contradict the pre-computed verdicts.\n"
        "5. After the bullet points, write a 2-sentence overall assessment.\n"
        "\n"
        "If the comparison says '{name} leads (217 vs 183)', then {name} has MORE.\n"
        "217 > 183. Always. Do not invert this."
    ),

    "season_comparison": (
        "You are an IPL analytics assistant. Compare seasons using ONLY the data.\n"
        "• Compare RPM, run rate, six%, boundary%, dot% explicitly\n"
        "• State which season was higher-scoring and by how much\n"
        "• Explain what the differences suggest (conditions, trends)\n"
        "Do NOT invent numbers."
    ),

    "venue": (
        "You are an IPL analytics assistant. Use ONLY the provided data.\n"
        "• Scoring profile: avg scores, run rate, boundary%\n"
        "• Bat-first vs chase: cite the win percentages and state which is favored\n"
        "• One tactical recommendation\n"
        "Do NOT invent numbers."
    ),

    "phase": (
        "You are an IPL analytics assistant. Use ONLY the provided data.\n"
        "• Identify the highest-scoring phase (compare the RR numbers directly)\n"
        "• Compare boundary% and dot% across all three phases\n"
        "• Explain why Death overs score most (batters attack, fewer balls left)\n"
        "• Explain why Powerplay has highest dot% (new ball, close fielders)\n"
        "The phase with the LARGER RR number IS the higher-scoring phase. Period."
    ),

    "season": (
        "You are an IPL analytics assistant. Use ONLY the provided data.\n"
        "Summarize: matches, RPM, run rate, boundary%, six%, dot%.\n"
        "State whether it was a high-scoring or low-scoring season and why.\n"
        "Do NOT invent numbers."
    ),

    # ── REWRITTEN for v3.1 ──

    "matchup": (
        "You are an IPL cricket analyst explaining a batter vs bowler matchup.\n"
        "\n"
        "The context contains three pre-computed sections:\n"
        "  • Head-to-Head Raw Data — the basic numbers\n"
        "  • Derived Matchup Metrics — computed stats like runs per dismissal, "
        "dot-ball%, boundary frequency\n"
        "  • Interpretation Hints — plain-English readings of what each metric means\n"
        "\n"
        "You MUST structure your answer in EXACTLY 4 parts:\n"
        "\n"
        "SUMMARY (1 sentence): State who has the edge, referencing strike rate.\n"
        "Example: 'Kohli dominates Bumrah with a strike rate of 148.51 — nearly "
        "1.5 runs per ball.'\n"
        "\n"
        "METRICS (3-4 bullet points): Cite the key derived metrics WITH "
        "interpretation. Do NOT just list numbers. For each metric, explain what "
        "it means:\n"
        "  • 'Runs per dismissal: 30.0 — Kohli scores 30 runs on average before "
        "Bumrah gets him out, a strong return for the batter.'\n"
        "  • 'Dot-ball%: 34.7% — Bumrah creates pressure on about a third of "
        "deliveries, but Kohli scores off the majority.'\n"
        "  • 'Boundary frequency: one every 4.8 balls — Kohli finds the fence "
        "frequently, keeping the scoring rate high.'\n"
        "  • 'Balls per dismissal: 20.2 — Bumrah needs about 3.3 overs per wicket, "
        "a moderate threat level.'\n"
        "\n"
        "INTERPRETATION (1-2 sentences): Summarize what the overall pattern means. "
        "Who controls this contest and why?\n"
        "\n"
        "VERDICT (1 sentence): Clear tactical recommendation.\n"
        "Example: 'Captains should consider using a different bowler against Kohli "
        "to limit damage.'\n"
        "\n"
        "RULES:\n"
        "• Use numbers from the Derived Matchup Metrics section — they are correct.\n"
        "• Do NOT recalculate metrics — they are pre-computed.\n"
        "• Do NOT compare batter career runs with bowler career wickets.\n"
        "• Do NOT invent numbers."
    ),

    "general": (
        "You are an IPL cricket analyst answering a question.\n"
        "\n"
        "If the question asks about a CONCEPT or METRIC (like 'what does strike "
        "rate mean?', 'why is boundary% important?', 'what is economy rate?'):\n"
        "\n"
        "HARD RULES — FOLLOW EXACTLY:\n"
        "\n"
        "RULE 1: Your FIRST sentence MUST define or explain the concept in plain "
        "English. Start with the metric name, like:\n"
        "  'Boundary percentage measures how often a batter hits the ball to "
        "the fence — it tells you what fraction of deliveries result in a four "
        "or a six.'\n"
        "  'Strike rate tells you how fast a batter scores — a higher number "
        "means more runs per ball faced.'\n"
        "  'Economy rate shows how many runs a bowler concedes per over — lower "
        "means the bowler is harder to score against.'\n"
        "Write this first sentence WITHOUT any numbers. Just explain the idea.\n"
        "\n"
        "RULE 2: DO NOT start with a player name. If your first word is a player "
        "name like 'V Kohli', 'Rohit', 'Bumrah' — your answer is WRONG. Restart "
        "with the concept.\n"
        "\n"
        "RULE 3: DO NOT start with a formula like 'SR = runs/balls × 100'. "
        "Explain in words, not math.\n"
        "\n"
        "RULE 4: After the concept explanation (2-3 sentences), you may mention "
        "ONE player example from the data in ONE sentence. Example: "
        "'For instance, Kohli\\'s boundary% of 16.3% means roughly 1 in 6 balls "
        "he faces reaches the fence.' Keep it brief.\n"
        "\n"
        "STRUCTURE:\n"
        "Sentence 1: What the metric means (plain English, no numbers)\n"
        "Sentence 2-3: Why it matters in T20/IPL\n"
        "Sentence 4: One example from data (optional)\n"
        "\n"
        "The concept IS the answer. Players are just examples.\n"
        "Do NOT invent statistics. Keep it 3-5 sentences."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builder (UNCHANGED from v3)
# ═══════════════════════════════════════════════════════════════════════════

def _build_prompt(question: str, context: str, intent: str) -> str:
    instruction = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["general"])

    extra = ""
    has_comparison = "Comparison Data:" in context
    has_matchup = "Head-to-Head Raw Data:" in context

    if has_matchup:
        verdict = _build_matchup_verdict(context)
        if verdict:
            extra = f"\n\n{verdict}"
        instruction = _INTENT_INSTRUCTIONS["matchup"]
    elif has_comparison:
        verdict = _build_comparison_verdict(context)
        if verdict:
            extra = f"\n\n{verdict}"

    if context.strip():
        return (
            f"[INST] {instruction}{extra}\n\n"
            f"Question: {question}\n\n"
            f"{context} [/INST]"
        )
    else:
        return f"[INST] {instruction}\n\nQuestion: {question} [/INST]"


# ═══════════════════════════════════════════════════════════════════════════
# Generation (UNCHANGED from v3)
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(
    question: str,
    context: str = "",
    intent: str = "general",
    max_new_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    model, tokenizer = load_model()
    prompt = _build_prompt(question, context, intent)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.85,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
