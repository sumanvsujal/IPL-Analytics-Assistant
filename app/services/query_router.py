"""
Query Router — Classifies queries and extracts entities
========================================================
Rule-based router. Determines route, intent, and entities.
"""

import re
from app.services.structured_retriever import (
    find_batter, find_bowler, find_venue,
    batter_career, bowler_career, _NAME_ALIASES,
)

_SEASON_PATTERN = re.compile(r'\b(20[012]\d)\b')
_PHASE_NAMES = {"powerplay", "middle overs", "death overs", "death", "middle"}

# Words that should never trigger player name matching
_STOP = {
    "the","a","an","is","are","was","were","what","who","how","why","when",
    "which","where","does","do","did","has","have","had","be","been","for",
    "in","on","at","to","from","with","and","or","but","not","if","of","it",
    "ipl","cricket","top","best","highest","most","lowest","run","runs","rate",
    "wicket","wickets","compare","match","matches","stats","season","scoring",
    "stadium","overs","boundary","ball","dot","average","economy","about",
    "tell","me","give","show","explain","important","percentage","head-to-head",
    "more","than","between","difference","vs","versus","that","this","all",
    "time","ever","history","career","overall","performance","scored","taken",
}

# Known PRIMARY bowlers — players whose main role is bowling.
# A player in both tables is a "primary bowler" if they have 50+ wickets
# and either <1000 batting runs or more wickets than batting runs/30.
_PRIMARY_BOWLER_NAMES = set()
_bc_names = set(batter_career()["batter"].values)
for _, row in bowler_career().iterrows():
    name = row["bowler"]
    if row["wickets"] < 30:
        continue
    # Check if they have significant batting stats
    bat_row = batter_career()[batter_career()["batter"] == name]
    if len(bat_row) == 0 or bat_row.iloc[0]["runs"] < 1000:
        _PRIMARY_BOWLER_NAMES.add(name)
    elif row["wickets"] > bat_row.iloc[0]["runs"] / 25:
        # More wickets than expected for a batter → primary bowler
        _PRIMARY_BOWLER_NAMES.add(name)

_KNOWN_BOWLER_ALIASES = {
    a for a, t in _NAME_ALIASES.items()
    if t in _PRIMARY_BOWLER_NAMES
}


def extract_entities(query: str) -> dict:
    """Extract player names, seasons, venues, phases from a query."""
    ents = {"batters": [], "bowlers": [], "seasons": [], "venues": [], "phases": []}

    # ── Seasons ──
    ents["seasons"] = [int(m) for m in _SEASON_PATTERN.findall(query)]

    # ── Phases ──
    ql = query.lower()
    for p in _PHASE_NAMES:
        if p in ql:
            canon = "Powerplay" if "power" in p else "Death" if "death" in p else "Middle"
            if canon not in ents["phases"]:
                ents["phases"].append(canon)

    # ── Venues (check before players — venues contain common words) ──
    words = query.split()
    for w in range(6, 1, -1):
        for i in range(len(words) - w + 1):
            candidate = " ".join(words[i:i+w]).strip(".,?!")
            v = find_venue(candidate)
            if v and v not in ents["venues"]:
                ents["venues"].append(v)

    # ── Players via aliases (most reliable) ──
    matched = set()
    bc_names = set(batter_career()["batter"].values)
    bw_names = set(bowler_career()["bowler"].values)

    for w in range(3, 0, -1):
        for i in range(len(words) - w + 1):
            candidate = " ".join(words[i:i+w]).strip(".,?!").lower()
            # Strip possessive 's
            if candidate.endswith("'s") or candidate.endswith("'s"):
                candidate = candidate[:-2]
            if candidate in _STOP or len(candidate) < 3:
                continue
            if candidate in _NAME_ALIASES:
                target = _NAME_ALIASES[candidate]
                if target in matched:
                    continue
                matched.add(target)
                # Classify as bowler first if they are primarily a bowler
                if target in bw_names and candidate in _KNOWN_BOWLER_ALIASES:
                    ents["bowlers"].append(target)
                elif target in bc_names:
                    ents["batters"].append(target)
                elif target in bw_names:
                    ents["bowlers"].append(target)

    # ── Players via exact full-name match (2-3 word windows) ──
    for w in range(3, 1, -1):
        for i in range(len(words) - w + 1):
            candidate = " ".join(words[i:i+w]).strip(".,?!")
            if candidate.endswith("'s") or candidate.endswith("'s"):
                candidate = candidate[:-2]
            if any(tok.lower() in _STOP for tok in candidate.split()):
                continue
            if len(candidate) < 5:
                continue
            cl = candidate.lower()
            for name in bc_names:
                if cl == name.lower() and name not in matched:
                    ents["batters"].append(name)
                    matched.add(name)
                    break
            for name in bw_names:
                if cl == name.lower() and name not in matched:
                    ents["bowlers"].append(name)
                    matched.add(name)
                    break

    return ents


def classify_query(query: str) -> dict:
    """Classify a query into route type, intent, and entities."""
    ql = query.lower()
    entities = extract_entities(query)

    batters  = entities["batters"]
    bowlers  = entities["bowlers"]
    seasons  = entities["seasons"]
    venues   = entities["venues"]
    phases   = entities["phases"]
    has_players = bool(batters or bowlers)

    # ── Keyword scores ──
    struct_kw = ["stats","record","runs","wickets","economy","strike rate",
                 "average","top","highest","most","best","lowest","scored",
                 "taken","how many","career","in ipl","head-to-head","matchup"]
    insight_kw = ["why","explain","what does","what is","how does","significance",
                  "important","indicate","imply","trend","pattern","evolve",
                  "change","reason","meaning"]
    compare_kw = ["compare","vs","versus","difference between","better"]
    leader_kw  = ["top","highest","most","best","lowest","leading","all-time"]
    phase_kw   = ["phase","powerplay","death over","middle over"]
    venue_kw   = ["stadium","ground","venue","chinnaswamy","wankhede","eden",
                  "chepauk","feroz","jaitley","sawai","narendra modi"]

    s_score = sum(1 for k in struct_kw if k in ql)
    i_score = sum(1 for k in insight_kw if k in ql)
    is_compare = any(k in ql for k in compare_kw)
    is_leader  = any(k in ql for k in leader_kw)
    is_phase   = any(k in ql for k in phase_kw)
    is_venue   = any(k in ql for k in venue_kw) or bool(venues)

    # ── Intent classification ──
    if is_compare and len(seasons) >= 2:
        intent = "season_comparison"
    elif is_compare and has_players:
        intent = "comparison"
    elif is_compare and len(seasons) >= 2:
        intent = "season_comparison"
    elif has_players and any(k in ql for k in ["vs","against","head-to-head","matchup"]):
        # Check if it's batter vs bowler matchup
        if batters and bowlers:
            intent = "matchup"
        else:
            intent = "comparison"
    elif is_leader and is_phase:
        intent = "phase"
    elif is_leader:
        intent = "leaderboard"
    elif is_venue:
        intent = "venue"
    elif is_phase or phases:
        intent = "phase"
    elif seasons and not has_players:
        intent = "season"
    elif has_players:
        intent = "player_lookup"
    else:
        intent = "general"

    # ── Route ──
    if i_score > s_score and intent == "general":
        route = "insight"
    elif i_score > 0 and (s_score > 0 or has_players):
        route = "mixed"
    elif s_score > 0 or has_players or venues or phases or seasons:
        route = "structured"
    elif i_score > 0:
        route = "insight"
    else:
        route = "mixed"

    return {
        "route": route,
        "intent": intent,
        "entities": entities,
        "is_comparison": is_compare,
    }
