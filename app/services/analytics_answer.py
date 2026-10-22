"""Offline answers assembled from retrieved facts; no model or network required."""
import re

_DEFINITIONS = {
    "boundary": "Boundary percentage measures how often a delivery produces a four or six. It helps describe attacking intent, but should be considered alongside dot balls and dismissals. A four and a six both count as one boundary delivery.",
    "strike rate": "Batting strike rate is runs scored per 100 balls faced. A higher value means faster scoring. Compare it alongside average and the number of balls faced: scoring quickly and avoiding dismissal are different strengths.",
    "economy": "Bowling economy measures runs conceded per over. A lower economy indicates better run containment. Wickets and the phase of the innings provide additional context.",
    "dot": "Dot-ball percentage measures the share of deliveries on which no runs are scored. More dots can build pressure, but boundaries and wickets also matter when assessing a performance.",
    "average": "Batting average measures runs per dismissal; bowling average measures runs conceded per wicket. Higher is generally better for batting, while lower is generally better for bowling. Small samples can be misleading.",
}


def generate_answer(query: str, context: str = "", intent: str = "general") -> str:
    """Present existing computed facts without inventing results or verdicts."""
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
