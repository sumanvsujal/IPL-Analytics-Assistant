"""
Orchestrator v2 — Intent-aware hybrid RAG pipeline
====================================================
Improvements over v1:
  - Passes intent + entities to context_builder for filtering
  - Controls insight injection: structured-only queries get 0-1 insights max
  - Passes intent to generator for template selection
  - Matchup queries always try both orderings
"""

from app.services.query_router import classify_query
from app.services import structured_retriever as sr
from app.services.insight_retriever import retrieve_insights
from app.services.context_builder import build_context
from app.services.generator import generate_answer


def _retrieve_structured(query_lower: str, route_info: dict) -> list[dict]:
    """Execute structured retrieval based on route classification."""
    intent   = route_info["intent"]
    entities = route_info["entities"]
    batters  = entities.get("batters", [])
    bowlers  = entities.get("bowlers", [])
    seasons  = entities.get("seasons", [])
    venues   = entities.get("venues", [])
    phases   = entities.get("phases", [])

    results = []

    if intent == "leaderboard":
        bowl_kw = any(k in query_lower for k in ["wicket","bowler","bowling","economy"])
        results = sr.get_top_bowlers(n=5) if bowl_kw else sr.get_top_batters(n=5)

    elif intent == "comparison":
        # 1 batter + 1 bowler with "vs" → matchup + career
        if batters and bowlers and any(k in query_lower for k in ["vs","head-to-head","against","matchup"]):
            m = sr.get_matchup(batters[0], bowlers[0]) or sr.get_matchup(bowlers[0], batters[0])
            if m: results.append(m)
            b = sr.get_batter_career(batters[0])
            w = sr.get_bowler_career(bowlers[0])
            if b: results.append(b)
            if w: results.append(w)
        elif len(bowlers) >= 2:
            for name in bowlers[:2]:
                r = sr.get_bowler_career(name)
                if r: results.append(r)
        elif len(batters) >= 2:
            for name in batters[:2]:
                r = sr.get_batter_career(name)
                if r: results.append(r)
        if not results:
            for p in batters:
                r = sr.get_batter_career(p)
                if r: results.append(r)
            for p in bowlers:
                r = sr.get_bowler_career(p)
                if r: results.append(r)

    elif intent == "season_comparison":
        for s in seasons:
            r = sr.get_season(s)
            if r: results.append(r)

    elif intent == "matchup":
        batter_name = batters[0] if batters else None
        bowler_name = bowlers[0] if bowlers else None
        if batter_name and bowler_name:
            m = sr.get_matchup(batter_name, bowler_name) or sr.get_matchup(bowler_name, batter_name)
            if m: results.append(m)

    elif intent == "venue":
        for v in venues:
            r = sr.get_venue(v)
            if r: results.append(r)
        if not results:
            words = query_lower.split()
            for w in range(5, 1, -1):
                for i in range(len(words) - w + 1):
                    v = sr.find_venue(" ".join(words[i:i+w]))
                    if v:
                        results.append(sr.get_venue(v))
                        break
                if results: break

    elif intent == "phase":
        if seasons:
            for s in seasons:
                results.extend(sr.get_phase_season(s))
        else:
            results = sr.get_phase_overall()

    elif intent == "season":
        for s in seasons:
            r = sr.get_season(s)
            if r: results.append(r)

    elif intent == "player_lookup":
        for name in batters:
            r = (sr.get_batter_season(name, seasons[0]) if seasons
                 else [sr.get_batter_career(name)])
            results.extend([x for x in r if x])
        for name in bowlers:
            r = (sr.get_bowler_season(name, seasons[0]) if seasons
                 else [sr.get_bowler_career(name)])
            results.extend([x for x in r if x])
        # Fallback: batter classified but is really a bowler
        for name in batters:
            if not any(r.get("entity") == name for r in results):
                r = sr.get_bowler_career(name)
                if r: results.append(r)
    else:
        for name in batters:
            r = sr.get_batter_career(name)
            if r: results.append(r)
        for name in bowlers:
            r = sr.get_bowler_career(name)
            if r: results.append(r)

    return results


def answer_query(query: str, verbose: bool = False) -> dict:
    """Full hybrid RAG pipeline with intent-aware context and generation."""
    ql = query.lower()
    route_info = classify_query(query)
    intent = route_info["intent"]
    entities = route_info["entities"]
    route = route_info["route"]

    # ── Retrieval ──
    structured = []
    insights = []

    if route in ("structured", "mixed"):
        structured = _retrieve_structured(ql, route_info)

    # Insight retrieval: controlled by route type
    if route == "insight":
        # Pure insight query — full insight retrieval
        insights = retrieve_insights(query, top_k=4)
    elif route == "mixed":
        # Mixed — get insights but they'll be filtered by context_builder
        insights = retrieve_insights(query, top_k=3)
    elif route == "structured" and not structured:
        # Structured route but nothing found — fallback to insights
        insights = retrieve_insights(query, top_k=3)
    # For successful structured queries: NO automatic insight injection.
    # The context_builder will selectively add 0-1 if relevant.

    # ── Context building (intent-aware) ──
    context = build_context(
        structured_results=structured,
        insight_results=insights,
        intent=intent,
        entities=entities,
    )

    # ── Generation (intent-aware) ──
    answer = generate_answer(query, context, intent=intent)

    result = {"answer": answer}
    if verbose:
        result["route"] = route_info
        result["context"] = context
        result["structured_count"] = len(structured)
        result["insight_count"] = len(insights)

    return result
