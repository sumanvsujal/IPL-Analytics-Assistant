#!/usr/bin/env python3
"""
run_hybrid_chat.py — End-to-end IPL Analytics Hybrid Chat
==========================================================
Interactive script that demonstrates the full pipeline:
  Query → Route → Retrieve → Context → Generate → Answer

Run from project root:
    python -m app.scripts.run_hybrid_chat
    python -m app.scripts.run_hybrid_chat --no-model    # skip LLM, show retrieval only
    python -m app.scripts.run_hybrid_chat --test         # run preset test queries
"""

import argparse
from app.services.orchestrator import answer_query, _retrieve_structured
from app.services.query_router import classify_query
from app.services.context_builder import build_context
from app.services.insight_retriever import retrieve_insights


TEST_QUERIES = [
    "Who are the top IPL run scorers?",
    "What are V Kohli's IPL career stats?",
    "Compare JJ Bumrah and YS Chahal.",
    "Why is boundary percentage important?",
    "Which phase has the highest run rate and why?",
    "Is SP Narine economical?",
    "What are the stats for M Chinnaswamy Stadium?",
    "Compare Kohli and Rohit.",
    "How did IPL 2024 compare to 2023?",
    "What is the head-to-head record between V Kohli and JJ Bumrah?",
]


def run_single_query(query: str, use_model: bool = True):
    """Run a single query through the full pipeline with verbose output."""
    print(f"\n{'━' * 70}")
    print(f"❓ QUERY: {query}")
    print(f"{'━' * 70}")

    # Route
    route = classify_query(query)
    print(f"\n🔀 Route: {route['route']}  |  Intent: {route['intent']}")
    if route["entities"]["batters"]:
        print(f"   Batters: {route['entities']['batters']}")
    if route["entities"]["bowlers"]:
        print(f"   Bowlers: {route['entities']['bowlers']}")
    if route["entities"]["seasons"]:
        print(f"   Seasons: {route['entities']['seasons']}")
    if route["entities"]["venues"]:
        print(f"   Venues: {route['entities']['venues']}")
    if route["entities"]["phases"]:
        print(f"   Phases: {route['entities']['phases']}")

    if use_model:
        result = answer_query(query, verbose=True)

        if result.get("context"):
            print(
                f"\n📋 CONTEXT "
                f"({result.get('structured_count', 0)} structured, "
                f"{result.get('insight_count', 0)} insights):"
            )
            ctx = result["context"]
            if len(ctx) > 500:
                print(f"   {ctx[:500]}…")
            else:
                print(f"   {ctx}")

        print("\n💬 ANSWER:")
        print(f"   {result['answer']}")
    else:
        # Retrieval-only mode
        structured = _retrieve_structured(query, route)
        insights = retrieve_insights(query, top_k=2)
        context = build_context(structured, insights)

        print("\n📋 RETRIEVED CONTEXT:")
        if context:
            print(f"   {context[:600]}{'…' if len(context) > 600 else ''}")
        else:
            print("   (no context retrieved)")


def main():
    parser = argparse.ArgumentParser(description="IPL Analytics Hybrid Chat")
    parser.add_argument("--test", action="store_true",
                        help="Run preset test queries")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip LLM loading — show retrieval only")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query")
    args = parser.parse_args()

    print("=" * 70)
    print("IPL ANALYTICS — HYBRID RAG ASSISTANT")
    print("=" * 70)

    if args.test:
        for q in TEST_QUERIES:
            run_single_query(q, use_model=not args.no_model)
        return

    if args.query:
        run_single_query(args.query, use_model=not args.no_model)
        return

    # Interactive mode
    print("\nType a cricket analytics question (or 'quit' to exit).\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        run_single_query(query, use_model=not args.no_model)

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
