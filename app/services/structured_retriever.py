"""
Structured Retriever — Exact fact retrieval from analytics tables
=================================================================
Uses pandas on parquet files. This is the source of truth for all numbers.
No vector search, no approximation — exact lookups and aggregations.

Supports:
  - Player career/season stats (batting + bowling)
  - Leaderboard queries (top N by metric)
  - Comparisons between two entities
  - Venue stats
  - Phase stats
  - Season summaries
  - Head-to-head matchups
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Analytics data directory
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "analytics"

# ── Load tables once at import ──────────────────────────────────────────

_TABLES = {}

def _load(name: str) -> pd.DataFrame:
    if name not in _TABLES:
        _TABLES[name] = pd.read_parquet(_DATA_DIR / f"{name}.parquet")
    return _TABLES[name]

def batter_career() -> pd.DataFrame: return _load("batter_career_stats")
def batter_season() -> pd.DataFrame: return _load("batter_season_stats")
def bowler_career() -> pd.DataFrame: return _load("bowler_career_stats")
def bowler_season() -> pd.DataFrame: return _load("bowler_season_stats")
def venue_stats()   -> pd.DataFrame: return _load("venue_stats")
def phase_stats()   -> pd.DataFrame: return _load("phase_season_stats")
def season_summary() -> pd.DataFrame: return _load("season_summary")
def matchup_stats() -> pd.DataFrame: return _load("matchup_stats")


# ── Entity name matching ────────────────────────────────────────────────

# Common name aliases → dataset name
_NAME_ALIASES = {
    "rohit": "RG Sharma", "rohit sharma": "RG Sharma",
    "kohli": "V Kohli", "virat": "V Kohli", "virat kohli": "V Kohli",
    "dhoni": "MS Dhoni", "ms dhoni": "MS Dhoni",
    "dhawan": "S Dhawan", "shikhar": "S Dhawan",
    "warner": "DA Warner", "david warner": "DA Warner",
    "gayle": "CH Gayle", "chris gayle": "CH Gayle",
    "abd": "AB de Villiers", "de villiers": "AB de Villiers", "ab de villiers": "AB de Villiers",
    "rahul": "KL Rahul", "kl rahul": "KL Rahul",
    "raina": "SK Raina", "suresh raina": "SK Raina",
    "buttler": "JC Buttler", "jos buttler": "JC Buttler",
    "russell": "AD Russell", "andre russell": "AD Russell",
    "narine": "SP Narine", "sunil narine": "SP Narine",
    "bumrah": "JJ Bumrah", "jasprit bumrah": "JJ Bumrah",
    "chahal": "YS Chahal", "yuzvendra chahal": "YS Chahal",
    "ashwin": "R Ashwin", "ravichandran ashwin": "R Ashwin",
    "malinga": "SL Malinga", "lasith malinga": "SL Malinga",
    "rashid": "Rashid Khan", "rashid khan": "Rashid Khan",
    "jadeja": "RA Jadeja", "ravindra jadeja": "RA Jadeja",
    "bravo": "DJ Bravo", "dwayne bravo": "DJ Bravo",
    "harbhajan": "Harbhajan Singh",
    "patel": "HV Patel", "harshal": "HV Patel", "harshal patel": "HV Patel",
    "bhuvneshwar": "B Kumar", "bhuvi": "B Kumar", "bhuvneshwar kumar": "B Kumar",
    "yadav": "SA Yadav", "surya": "SA Yadav", "suryakumar": "SA Yadav", "sky": "SA Yadav",
    "samson": "SV Samson", "sanju samson": "SV Samson",
    "du plessis": "F du Plessis", "faf": "F du Plessis",
    "gambhir": "G Gambhir", "gautam gambhir": "G Gambhir",
    "watson": "SR Watson", "shane watson": "SR Watson",
}

def _fuzzy_match(query: str, candidates: pd.Series) -> str | None:
    """Case-insensitive match against a column of names, with alias support."""
    q = query.strip().lower()
    # 1. Check aliases first
    if q in _NAME_ALIASES:
        alias_target = _NAME_ALIASES[q]
        if alias_target in candidates.values:
            return alias_target
    # 2. Exact match
    for name in candidates:
        if q == name.lower():
            return name
    # 3. Substring match
    for name in candidates:
        if q in name.lower() or name.lower() in q:
            return name
    # 4. Last-name match
    for name in candidates:
        parts = name.lower().split()
        if any(q == p for p in parts):
            return name
    return None


def find_batter(name: str) -> str | None:
    return _fuzzy_match(name, batter_career()["batter"])

def find_bowler(name: str) -> str | None:
    return _fuzzy_match(name, bowler_career()["bowler"])

def find_venue(name: str) -> str | None:
    return _fuzzy_match(name, venue_stats()["venue"])


# ── Query functions ─────────────────────────────────────────────────────

def get_batter_career(name: str) -> dict | None:
    """Look up a batter's career stats."""
    matched = find_batter(name)
    if not matched:
        return None
    row = batter_career()[batter_career()["batter"] == matched].iloc[0]
    return {"entity": matched, "type": "batter_career", "data": row.to_dict()}


def get_batter_season(name: str, season: int = None) -> list[dict]:
    """Look up a batter's season stats. If no season specified, return best."""
    matched = find_batter(name)
    if not matched:
        return []
    df = batter_season()[batter_season()["batter"] == matched]
    if season:
        df = df[df["season"] == season]
    return [{"entity": matched, "season": int(r["season"]), "type": "batter_season",
             "data": r.to_dict()} for _, r in df.iterrows()]


def get_bowler_career(name: str) -> dict | None:
    """Look up a bowler's career stats."""
    matched = find_bowler(name)
    if not matched:
        return None
    row = bowler_career()[bowler_career()["bowler"] == matched].iloc[0]
    return {"entity": matched, "type": "bowler_career", "data": row.to_dict()}


def get_bowler_season(name: str, season: int = None) -> list[dict]:
    matched = find_bowler(name)
    if not matched:
        return []
    df = bowler_season()[bowler_season()["bowler"] == matched]
    if season:
        df = df[df["season"] == season]
    return [{"entity": matched, "season": int(r["season"]), "type": "bowler_season",
             "data": r.to_dict()} for _, r in df.iterrows()]


def get_top_batters(metric: str = "runs", n: int = 5, min_runs: int = 0) -> list[dict]:
    """Get top N batters by a given metric."""
    df = batter_career()
    if min_runs:
        df = df[df["runs"] >= min_runs]
    if metric not in df.columns:
        return []
    top = df.nlargest(n, metric)
    return [{"rank": i+1, "type": "batter_leaderboard", "metric": metric,
             "data": r.to_dict()} for i, (_, r) in enumerate(top.iterrows())]


def get_top_bowlers(metric: str = "wickets", n: int = 5, min_wickets: int = 0) -> list[dict]:
    df = bowler_career()
    if min_wickets:
        df = df[df["wickets"] >= min_wickets]
    if metric not in df.columns:
        return []
    if metric == "economy":
        top = df[df["balls_bowled"] >= 300].nsmallest(n, metric)
    else:
        top = df.nlargest(n, metric)
    return [{"rank": i+1, "type": "bowler_leaderboard", "metric": metric,
             "data": r.to_dict()} for i, (_, r) in enumerate(top.iterrows())]


def get_venue(name: str) -> dict | None:
    matched = find_venue(name)
    if not matched:
        return None
    row = venue_stats()[venue_stats()["venue"] == matched].iloc[0]
    return {"entity": matched, "type": "venue", "data": row.to_dict()}


def get_phase_overall() -> list[dict]:
    """Get aggregated phase stats across all seasons."""
    ps = phase_stats()
    agg = ps.groupby("match_phase").agg({
        "total_balls": "sum", "total_runs": "sum", "wickets": "sum",
        "boundaries": "sum", "dot_balls": "sum", "sixes": "sum",
    }).reset_index()
    agg["run_rate"] = (agg["total_runs"] / (agg["total_balls"] / 6)).round(2)
    agg["boundary_pct"] = (agg["boundaries"] / agg["total_balls"] * 100).round(2)
    agg["dot_pct"] = (agg["dot_balls"] / agg["total_balls"] * 100).round(2)
    agg["six_pct"] = (agg["sixes"] / agg["total_balls"] * 100).round(2)
    agg["wkts_per_over"] = (agg["wickets"] / (agg["total_balls"] / 6)).round(3)
    return [{"phase": r["match_phase"], "type": "phase_overall",
             "data": r.to_dict()} for _, r in agg.iterrows()]


def get_phase_season(season: int) -> list[dict]:
    ps = phase_stats()
    df = ps[ps["season"] == season]
    return [{"phase": r["match_phase"], "season": int(r["season"]),
             "type": "phase_season", "data": r.to_dict()} for _, r in df.iterrows()]


def get_season(season: int) -> dict | None:
    df = season_summary()
    row = df[df["season"] == season]
    if len(row) == 0:
        return None
    return {"season": int(season), "type": "season_summary",
            "data": row.iloc[0].to_dict()}


def get_matchup(batter_name: str, bowler_name: str) -> dict | None:
    batter = find_batter(batter_name)
    bowler = find_bowler(bowler_name)
    if not batter or not bowler:
        return None
    df = matchup_stats()
    row = df[(df["batter"] == batter) & (df["bowler"] == bowler)]
    if len(row) == 0:
        return None
    return {"batter": batter, "bowler": bowler, "type": "matchup",
            "data": row.iloc[0].to_dict()}


def compare_batters(name1: str, name2: str) -> list[dict]:
    """Retrieve career stats for two batters for comparison."""
    results = []
    for name in [name1, name2]:
        r = get_batter_career(name)
        if r:
            results.append(r)
    return results


def compare_bowlers(name1: str, name2: str) -> list[dict]:
    results = []
    for name in [name1, name2]:
        r = get_bowler_career(name)
        if r:
            results.append(r)
    return results
