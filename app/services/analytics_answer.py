"""Offline answers assembled from retrieved facts; no model or network required."""
import re
import math

_DEFINITIONS = {
    "boundary": "Boundary percentage measures how often a delivery produces a four or six. It helps describe attacking intent, but should be considered alongside dot balls and dismissals. A four and a six both count as one boundary delivery.",
    "strike rate": "Batting strike rate is runs scored per 100 balls faced. A higher value means faster scoring. Compare it alongside average and the number of balls faced: scoring quickly and avoiding dismissal are different strengths.",
    "economy": "Bowling economy measures runs conceded per over. A lower economy indicates better run containment. Wickets and the phase of the innings provide additional context.",
    "dot": "Dot-ball percentage measures the share of deliveries on which no runs are scored. More dots can build pressure, but boundaries and wickets also matter when assessing a performance.",
    "average": "Batting average measures runs per dismissal; bowling average measures runs conceded per wicket. Higher is generally better for batting, while lower is generally better for bowling. Small samples can be misleading.",
}


def generate_answer(query: str, context: str = "", intent: str = "general", structured=None, entities=None) -> str:
    """Present existing computed facts without inventing results or verdicts."""
    analysis = _explain(query, structured or [], entities or {})
    if analysis:
        return analysis + "\n\nSource: bundled historical IPL data. Live results are not included."
    explanation = ""
    if intent in ("insight", "explanation", "general"):
        for term, text in _DEFINITIONS.items():
            if term in query.lower():
                explanation = text
                break

    # Context includes heuristic interpretation hints aimed at the optional LLM.
    # Show measured facts, not those uncalibrated winner/captain recommendations.
    sections = []
    for section in context.split("\n\n"):
        if section.startswith("Interpretation Hints:"):
            continue
        lines = section.splitlines()
        if lines and lines[0] in {"Data:", "Comparison Data:", "Background:",
                                   "Head-to-Head Raw Data:", "Career Context:",
                                   "Derived Matchup Metrics:"}:
            lines = lines[1:]
        lines = [re.sub(r"^Player [AB]:\s*", "", line) for line in lines]
        if lines:
            sections.append("\n\n".join(lines))

    if not sections and not explanation:
        return ("I couldn't find relevant facts in the bundled IPL dataset. "
                "Try a named player, venue, season, or a question about strike rate, "
                "economy, or boundary percentage. This demo cannot answer live scores or general questions.")

    parts = [explanation] if explanation else []
    if sections:
        parts.append("From the bundled IPL analytics dataset:\n\n" + "\n\n".join(sections))
    if intent in ("comparison", "matchup"):
        parts.append("These figures describe the recorded sample; they do not predict the next match.")
    parts.append("Source: bundled historical IPL data. Live results are not included.")
    return "\n\n".join(parts)


def _number(data, key):
    value = data.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _edge(a, b, key, label, lower=False, unit=""):
    x, y = _number(a, key), _number(b, key)
    if x is None or y is None:
        return None
    an, bn = a.get("batter", a.get("bowler")), b.get("batter", b.get("bowler"))
    value_unit = "%" if unit == " percentage points" else unit
    if abs(x - y) < .005:
        return f"**{label}:** effectively level ({x:.2f}{value_unit} vs {y:.2f}{value_unit})."
    name = an if (x < y if lower else x > y) else bn
    return f"**{label}: {name} leads** — {an} {x:.2f}{value_unit} vs {bn} {y:.2f}{value_unit}, a gap of {abs(x-y):.2f}{unit}."


def _explain(query, rows, entities):
    """Derive an explanation from typed records, never from guessed text values."""
    scope = ""
    if entities.get("phases") and any(r["type"] in {"batter_career", "bowler_career"} for r in rows):
        scope = ("**Scope limitation:** You asked about " + ", ".join(entities["phases"]) +
                 ". The bundled tables do not contain player-by-phase records, so I cannot "
                 "determine who is better in that phase. The following is an overall career comparison only.\n\n")
    elif entities.get("seasons") and any(r["type"] in {"batter_career", "bowler_career"} for r in rows):
        scope = "**Scope limitation:** These are career totals, not the requested season-specific comparison.\n\n"

    for row in rows:
        if row["type"] == "matchup":
            d = row["data"]
            balls, runs, wickets = (int(d[k]) for k in ("balls", "runs", "wickets"))
            sr = float(d["strike_rate"])
            answer = (f"**{d['batter']} vs {d['bowler']}: scoring rate and dismissal risk tell different stories.**\n\n"
                      f"{d['batter']} scored **{runs} runs from {balls} balls**, a strike rate of **{sr:.2f}** "
                      f"({sr / 100:.2f} runs per ball). {d['bowler']} took their wicket **{wickets} times** in this recorded matchup.")
            if wickets:
                answer += f" That is {runs / wickets:.2f} runs and {balls / wickets:.2f} balls per dismissal."
            else:
                answer += " There were no recorded dismissals; a runs-per-dismissal average cannot be calculated."
            return scope + answer + "\n\nUse both scoring speed and dismissals to judge the matchup; these historical figures alone do not establish who will win their next encounter."

    players = [r["data"] for r in rows if r["type"] in {"batter_career", "bowler_career", "batter_season", "bowler_season"}]
    if len(players) != 2 or ("batter" in players[0]) != ("batter" in players[1]):
        return None
    a, b = players
    batting = "batter" in a
    name_key = "batter" if batting else "bowler"
    q = query.lower()
    metrics = ([("runs", "Run accumulation", False, ""), ("strike_rate", "Scoring speed", False, ""),
                ("batting_average", "Runs per dismissal", False, ""), ("boundary_pct", "Boundary-ball percentage", False, " percentage points"),
                ("dot_pct", "Avoiding dot balls", True, " percentage points")]
               if batting else [("wickets", "Wicket tally", False, ""), ("economy", "Run containment", True, ""),
                                ("bowling_average", "Runs conceded per wicket", True, ""),
                                ("bowling_strike_rate", "Balls per wicket", True, "")])
    focus = next((key for terms, key in [(('strike rate', 'scoring speed', 'faster'), 'strike_rate' if batting else 'bowling_strike_rate'),
                                        (('economy', 'economical'), 'economy'), (('average',), 'batting_average' if batting else 'bowling_average'),
                                        (('boundary', 'boundaries'), 'boundary_pct'), (('dot',), 'dot_pct')]
                  if any(term in q for term in terms)), None)
    metrics.sort(key=lambda m: m[0] != focus)
    lines = [text for key, label, lower, unit in metrics if (text := _edge(a, b, key, label, lower, unit))]
    if not lines:
        return None
    headline = f"**{a[name_key]} vs {b[name_key]} — what separates them?**"
    if focus and _number(a, focus) is not None and _number(b, focus) is not None:
        headline = lines.pop(0)
    interpretation = ("Strike rate describes scoring speed; average describes runs per dismissal. "
                      "Boundary percentage counts deliveries hit for four or six, while a lower dot-ball percentage means fewer scoreless deliveries. "
                      "A player can lead on boundaries without leading on overall scoring speed."
                      if batting else "A lower economy means fewer runs conceded per over. A lower bowling average means fewer runs conceded per wicket, "
                      "and a lower bowling strike rate means wickets arrive more frequently. Wicket totals also depend on opportunities.")
    if batting and all(_number(d, key) is not None for d in (a, b) for key in ("strike_rate", "batting_average")):
        speed_gap = abs(float(a["strike_rate"]) - float(b["strike_rate"]))
        if speed_gap < 1:
            interpretation = (f"Their overall scoring speeds are very close: the difference is only {speed_gap:.2f} runs per 100 balls. " + interpretation)
    sample = f"Recorded sample: {a[name_key]} {int(a['matches'])} matches; {b[name_key]} {int(b['matches'])} matches."
    return scope + headline + "\n\n" + "\n\n".join(lines) + "\n\n**How to read this:** " + interpretation + "\n\n" + sample + " Career aggregates do not isolate batting position, venue, or opposition, and cannot establish an overall winner for every situation."
