"""
Context Builder v2 — Intent-aware context assembly
====================================================
Improvements over v1:
  - Intent-specific context templates (matchup-first, comparison side-by-side, etc.)
  - Insight filtering: only includes insights that are relevant to the query entities
  - Structured data always comes before insights
  - Comparison context explicitly labels Player A vs Player B
"""

import numpy as np


def _safe(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.2f}" if val != int(val) else str(int(val))
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════
# Individual formatters (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════

def format_batter_career(data: dict) -> str:
    d = data["data"]
    return (
        f"{d['batter']}: {int(d['runs'])} runs in {int(d['matches'])} matches "
        f"({int(d['seasons'])} seasons). SR {_safe(d['strike_rate'])}, avg {_safe(d.get('batting_average'))}. "
        f"{int(d['fours'])} fours, {int(d['sixes'])} sixes. "
        f"Boundary% {_safe(d['boundary_pct'])}%, dot% {_safe(d['dot_pct'])}%. "
        f"Highest: {int(d['highest_score'])}."
    )

def format_bowler_career(data: dict) -> str:
    d = data["data"]
    return (
        f"{d['bowler']}: {int(d['wickets'])} wickets in {int(d['matches'])} matches "
        f"({int(d['seasons'])} seasons). Economy {_safe(d['economy'])}, "
        f"avg {_safe(d.get('bowling_average'))}, SR {_safe(d.get('bowling_strike_rate'))}. "
        f"Dot% {_safe(d['dot_pct'])}%. Overs: {_safe(d['overs_bowled'])}."
    )

def format_venue(data: dict) -> str:
    d = data["data"]
    return (
        f"{d['venue']}: {int(d['matches'])} matches. "
        f"Avg 1st innings: {_safe(d['avg_first_innings_score'])}, "
        f"2nd: {_safe(d['avg_second_innings_score'])}. "
        f"RR {_safe(d['avg_run_rate'])}, boundary% {_safe(d['boundary_pct'])}%. "
        f"Bat-first win% {_safe(d['bat_first_win_pct'])}%, "
        f"chase win% {_safe(d['chase_win_pct'])}%."
    )

def format_phase(data: dict) -> str:
    d = data["data"]
    rr  = d.get('run_rate') or d.get('rr')
    bp  = d.get('boundary_pct') or d.get('bp')
    dp  = d.get('dot_pct') or d.get('dp')
    sp  = d.get('six_pct') or d.get('sp')
    wpo = d.get('wickets_per_over') or d.get('wkts_per_over') or d.get('wpo')
    return (
        f"{d['match_phase']}: RR {_safe(rr)}, boundary% {_safe(bp)}%, "
        f"dot% {_safe(dp)}%, six% {_safe(sp)}%, wkts/over {_safe(wpo)}."
    )

def format_season(data: dict) -> str:
    d = data["data"]
    return (
        f"IPL {int(d['season'])}: {int(d['matches'])} matches, "
        f"{int(d['runs_per_match'])} runs/match, RR {_safe(d['avg_run_rate'])}. "
        f"Six% {_safe(d['six_pct'])}%, boundary% {_safe(d['boundary_pct'])}%, "
        f"dot% {_safe(d['dot_pct'])}%."
    )

def format_matchup(data: dict) -> str:
    d = data["data"]
    return (
        f"{d['batter']} vs {d['bowler']}: {int(d['balls'])} balls, "
        f"{int(d['runs'])} runs (SR {_safe(d['strike_rate'])}). "
        f"Dismissed {int(d['wickets'])} times. "
        f"{int(d['fours'])} fours, {int(d['sixes'])} sixes, {int(d['dots'])} dots."
    )

def format_batter_season(data: dict) -> str:
    d = data["data"]
    return (
        f"{d['batter']} in IPL {int(d['season'])}: {int(d['runs'])} runs "
        f"({int(d['matches'])} matches), SR {_safe(d['strike_rate'])}, "
        f"avg {_safe(d.get('batting_average'))}. "
        f"{int(d['fours'])} fours, {int(d['sixes'])} sixes. Highest: {int(d['highest_score'])}."
    )

def format_leaderboard(results: list[dict]) -> str:
    lines = []
    for r in results:
        d = r["data"]
        if "batter" in d:
            lines.append(
                f"#{r['rank']} {d['batter']}: {int(d['runs'])} runs, "
                f"SR {_safe(d['strike_rate'])}, avg {_safe(d.get('batting_average'))}"
            )
        elif "bowler" in d:
            lines.append(
                f"#{r['rank']} {d['bowler']}: {int(d['wickets'])} wkts, "
                f"eco {_safe(d['economy'])}, avg {_safe(d.get('bowling_average'))}"
            )
    return "\n".join(lines)


_FORMATTERS = {
    "batter_career": format_batter_career,
    "bowler_career": format_bowler_career,
    "batter_season": format_batter_season,
    "bowler_season": lambda d: (
        f"{d['data']['bowler']} in IPL {int(d['data']['season'])}: "
        f"{int(d['data']['wickets'])} wkts, eco {_safe(d['data']['economy'])}, "
        f"dot% {_safe(d['data']['dot_pct'])}%"
    ),
    "venue": format_venue,
    "phase_overall": format_phase,
    "phase_season": format_phase,
    "season_summary": format_season,
    "matchup": format_matchup,
    "batter_leaderboard": None,
    "bowler_leaderboard": None,
}


# ═══════════════════════════════════════════════════════════════════════════
# Insight relevance filtering
# ═══════════════════════════════════════════════════════════════════════════

def _filter_insights(insights: list[dict], entities: dict, intent: str) -> list[dict]:
    """
    Filter insights to only those relevant to the query's entities/intent.
    Returns at most 2 insights, ranked by relevance.
    """
    if not insights:
        return []

    entity_names = set()
    for key in ["batters", "bowlers"]:
        entity_names.update(entities.get(key, []))
    for v in entities.get("venues", []):
        entity_names.add(v)

    scored = []
    for ins in insights:
        score = ins.get("score", 0)
        text_lower = ins["insight"].lower()

        # Boost if insight mentions a query entity
        entity_boost = 0
        for name in entity_names:
            if name.lower() in text_lower:
                entity_boost += 2

        # Boost if insight category matches intent
        cat = ins.get("category", "")
        category_boost = 0
        if intent == "phase" and "phase" in cat:
            category_boost = 1
        elif intent in ("player_lookup", "comparison") and "batting" in cat:
            category_boost = 0.5
        elif intent == "venue" and "venue" in cat:
            category_boost = 1
        elif intent in ("season", "season_comparison") and "season" in cat:
            category_boost = 1

        total = score + entity_boost + category_boost
        scored.append((total, ins))

    scored.sort(key=lambda x: -x[0])

    # Only return insights with meaningful relevance
    # For structured-heavy intents, be very selective
    if intent in ("leaderboard", "player_lookup", "matchup", "venue", "season"):
        threshold = 1.0
        max_count = 1
    else:
        threshold = 0.3
        max_count = 2

    return [ins for total, ins in scored[:max_count] if total >= threshold]


# ═══════════════════════════════════════════════════════════════════════════
# Intent-specific context templates
# ═══════════════════════════════════════════════════════════════════════════

def _build_comparison_context(structured: list[dict]) -> str:
    """Build side-by-side comparison context with explicit labels."""
    if len(structured) < 2:
        return _build_generic_structured(structured)

    lines = []
    for i, r in enumerate(structured):
        label = "Player A" if i == 0 else "Player B"
        rtype = r.get("type", "")
        formatter = _FORMATTERS.get(rtype)
        if formatter:
            lines.append(f"{label}: {formatter(r)}")
        else:
            lines.append(f"{label}: {r.get('data', r)}")

    return "Comparison Data:\n" + "\n".join(lines)


def _build_matchup_context(structured: list[dict]) -> str:
    """
    Build matchup context with three sections:
      1. Head-to-Head Raw Data — the basic matchup stats
      2. Derived Matchup Metrics — computed deterministically, NOT by the LLM
      3. Interpretation Hints — plain-English readings the LLM can use
    Career stats go last as secondary context.
    """
    matchup_data = None
    career_lines = []

    for r in structured:
        rtype = r.get("type", "")
        if rtype == "matchup":
            matchup_data = r.get("data", {})
        else:
            formatter = _FORMATTERS.get(rtype)
            text = formatter(r) if formatter else str(r.get("data", r))
            career_lines.append(text)

    if not matchup_data:
        # No matchup found — fall back to plain listing
        return _build_generic_structured(structured)

    # ── Extract raw values ──
    batter  = matchup_data.get("batter", "Batter")
    bowler  = matchup_data.get("bowler", "Bowler")
    balls   = int(matchup_data.get("balls", 0))
    runs    = int(matchup_data.get("runs", 0))
    wkts    = int(matchup_data.get("wickets", 0))
    dots    = int(matchup_data.get("dots", 0))
    fours   = int(matchup_data.get("fours", 0))
    sixes   = int(matchup_data.get("sixes", 0))
    sr      = float(matchup_data.get("strike_rate", 0))

    # ── Compute derived metrics deterministically ──
    boundaries      = fours + sixes
    runs_per_dismissal  = round(runs / wkts, 2) if wkts > 0 else None
    balls_per_dismissal = round(balls / wkts, 1) if wkts > 0 else None
    dot_pct             = round(dots / balls * 100, 1) if balls > 0 else 0.0
    boundary_pct        = round(boundaries / balls * 100, 1) if balls > 0 else 0.0
    balls_per_boundary  = round(balls / boundaries, 1) if boundaries > 0 else None
    scoring_balls       = balls - dots
    scoring_pct         = round(scoring_balls / balls * 100, 1) if balls > 0 else 0.0

    # ── Section 1: Raw Data ──
    raw = (
        f"Head-to-Head Raw Data:\n"
        f"{batter} vs {bowler}: {balls} balls, {runs} runs, "
        f"{wkts} dismissals, {fours} fours, {sixes} sixes, {dots} dots."
    )

    # ── Section 2: Derived Metrics ──
    derived_lines = [
        f"Derived Matchup Metrics:",
        f"• Strike rate: {sr:.2f} (batter scores {sr/100:.2f} runs per ball)",
        f"• Dot-ball%: {dot_pct}% ({dots} scoreless out of {balls} deliveries)",
        f"• Scoring-ball%: {scoring_pct}% ({scoring_balls} balls produced runs)",
        f"• Boundary%: {boundary_pct}% ({boundaries} boundaries in {balls} balls)",
    ]
    if balls_per_boundary is not None:
        derived_lines.append(f"• Boundary frequency: one boundary every {balls_per_boundary} balls")
    if runs_per_dismissal is not None:
        derived_lines.append(f"• Runs per dismissal: {runs_per_dismissal} (batter scores this many before getting out)")
    else:
        derived_lines.append(f"• Runs per dismissal: never dismissed (batter unbeaten)")
    if balls_per_dismissal is not None:
        derived_lines.append(f"• Balls per dismissal: {balls_per_dismissal} (bowler needs this many balls for a wicket)")
    else:
        derived_lines.append(f"• Balls per dismissal: N/A (no dismissals)")
    derived = "\n".join(derived_lines)

    # ── Section 3: Interpretation Hints ──
    hints = [f"Interpretation Hints:"]

    # SR verdict
    if sr > 140:
        hints.append(f"• SR {sr:.2f} is well above T20 par (~130): {batter} DOMINATES this matchup.")
    elif sr > 120:
        hints.append(f"• SR {sr:.2f} is above average: {batter} has the edge but {bowler} competes.")
    elif sr > 100:
        hints.append(f"• SR {sr:.2f} is below par: {bowler} restricts {batter} effectively.")
    else:
        hints.append(f"• SR {sr:.2f} is very low: {bowler} DOMINATES this matchup.")

    # Dismissal verdict
    if wkts == 0:
        hints.append(f"• {bowler} has NEVER dismissed {batter} — no breakthrough found.")
    elif balls_per_dismissal and balls_per_dismissal < 15:
        hints.append(f"• A wicket every {balls_per_dismissal} balls is VERY frequent — {bowler} is a major threat.")
    elif balls_per_dismissal and balls_per_dismissal < 25:
        hints.append(f"• A wicket every {balls_per_dismissal} balls is moderate — {bowler} can break through.")
    else:
        hints.append(f"• A wicket only every {balls_per_dismissal} balls — {batter} rarely gets out to {bowler}.")

    # Dot pressure verdict
    if dot_pct > 40:
        hints.append(f"• {dot_pct}% dots: {bowler} creates heavy pressure.")
    elif dot_pct > 30:
        hints.append(f"• {dot_pct}% dots: moderate pressure, but {batter} finds gaps.")
    else:
        hints.append(f"• {dot_pct}% dots: {batter} scores freely with little containment.")

    # Boundary verdict
    if balls_per_boundary and balls_per_boundary < 5:
        hints.append(f"• A boundary every {balls_per_boundary} balls — {batter} clears the fence frequently.")
    elif balls_per_boundary and balls_per_boundary < 8:
        hints.append(f"• A boundary every {balls_per_boundary} balls — decent hitting.")
    elif balls_per_boundary:
        hints.append(f"• A boundary only every {balls_per_boundary} balls — limited big shots.")

    # Overall
    if sr > 130:
        hints.append(f"• VERDICT: {batter} has the clear edge. Captains should avoid this matchup.")
    elif sr > 100:
        hints.append(f"• VERDICT: Slight edge to {bowler}. A competitive matchup.")
    else:
        hints.append(f"• VERDICT: {bowler} dominates. Captains should target this matchup.")

    interp = "\n".join(hints)

    # ── Assemble ──
    parts = [raw, derived, interp]
    if career_lines:
        parts.append("Career Context:\n" + "\n".join(career_lines))

    return "\n\n".join(parts)


def _build_generic_structured(structured: list[dict]) -> str:
    """Default structured formatting."""
    if not structured:
        return ""

    if structured[0].get("type", "").endswith("leaderboard"):
        return "Data:\n" + format_leaderboard(structured)

    lines = []
    for r in structured:
        rtype = r.get("type", "")
        formatter = _FORMATTERS.get(rtype)
        if formatter:
            lines.append(formatter(r))
        else:
            lines.append(str(r.get("data", r)))
    return "Data:\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_context(
    structured_results: list[dict] = None,
    insight_results: list[dict] = None,
    intent: str = "general",
    entities: dict = None,
) -> str:
    """
    Build intent-aware context string from retrieval results.

    Args:
        structured_results: from structured_retriever
        insight_results: from insight_retriever
        intent: query intent from router
        entities: extracted entities from router

    Returns:
        Formatted context string for the LLM.
    """
    structured_results = structured_results or []
    insight_results = insight_results or []
    entities = entities or {}

    parts = []

    # ── Structured section (always first) ──
    if structured_results:
        has_matchup = any(r.get("type") == "matchup" for r in structured_results)

        if intent == "comparison" and not has_matchup:
            parts.append(_build_comparison_context(structured_results))
        elif intent == "comparison" and has_matchup:
            parts.append(_build_matchup_context(structured_results))
        elif intent == "matchup":
            parts.append(_build_matchup_context(structured_results))
        else:
            parts.append(_build_generic_structured(structured_results))

    # ── Filtered insights (only if relevant) ──
    filtered = _filter_insights(insight_results, entities, intent)
    if filtered:
        lines = [r["insight"] for r in filtered]
        parts.append("Background:\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else ""
