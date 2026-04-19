#!/usr/bin/env python3
"""
test_answer_quality.py — Answer quality regression test
=========================================================
Runs 14 representative queries covering all intents and prints:
  - Route / intent classification
  - Context summary (what was retrieved)
  - Final answer

Run from project root:
    python -m app.scripts.test_answer_quality              # with LLM
    python -m app.scripts.test_answer_quality --no-model   # retrieval + context only
"""

import argparse, sys, types

TEST_QUERIES = [
    # Leaderboard
    ("Who are the top IPL run scorers?", "leaderboard"),
    ("Who are the top IPL wicket takers?", "leaderboard"),
    # Player lookup
    ("What are V Kohli's IPL career stats?", "player_lookup"),
    ("Is SP Narine economical?", "player_lookup"),
    ("What are MS Dhoni career stats?", "player_lookup"),
    # Comparison
    ("Compare Kohli and Rohit.", "comparison"),
    ("Compare Bumrah and Chahal.", "comparison"),
    # Matchup
    ("V Kohli vs JJ Bumrah head-to-head", "comparison/matchup"),
    # Venue
    ("What are the stats for Chinnaswamy Stadium?", "venue"),
    # Phase
    ("Which phase has the highest run rate?", "phase"),
    # Season comparison
    ("How did IPL 2024 compare to 2023?", "season_comparison"),
    # Insight / explanation
    ("Why is boundary percentage important?", "insight"),
    ("What does strike rate mean?", "explanation"),
    # Edge case: player with both bat/bowl records
    ("Tell me about Narine's IPL bowling.", "player_lookup"),
]


def run_tests(use_model: bool = True):
    # Mock generator if --no-model
    if not use_model:
        mock = types.ModuleType("app.services.generator")
        mock.generate_answer = lambda q, c="", intent="general": "[LLM SKIPPED — showing retrieval only]"
        mock.load_model = lambda **kw: (None, None)
        mock.is_finetuned = lambda: False
        sys.modules["app.services.generator"] = mock

    from app.services.orchestrator import answer_query
    from app.services.query_router import classify_query

    print("=" * 80)
    print("ANSWER QUALITY REGRESSION TEST")
    print("=" * 80)

    for i, (query, expected_type) in enumerate(TEST_QUERIES):
        route = classify_query(query)
        result = answer_query(query, verbose=True)

        print(f"\n{'━' * 80}")
        print(f"#{i+1:2d} [{expected_type}]")
        print(f"    Q: {query}")
        print(f"    Route: {route['route']}/{route['intent']}")

        ents = {k: v for k, v in route["entities"].items() if v}
        if ents:
            print(f"    Entities: {ents}")

        print(f"    Retrieved: {result.get('structured_count', 0)} structured, "
              f"{result.get('insight_count', 0)} insights")

        ctx = result.get("context", "")
        if ctx:
            # Show first 200 chars of context
            preview = ctx.replace("\n", " | ")[:200]
            print(f"    Context: {preview}{'…' if len(ctx) > 200 else ''}")

        answer = result["answer"]
        print(f"    Answer: {answer[:300]}{'…' if len(answer) > 300 else ''}")

    print(f"\n{'━' * 80}")
    print(f"✅ {len(TEST_QUERIES)} test queries completed")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model", action="store_true",
                        help="Skip LLM — show retrieval and context only")
    args = parser.parse_args()
    run_tests(use_model=not args.no_model)


if __name__ == "__main__":
    main()
