import os
import sys
import unittest
from unittest.mock import patch

from app.api.main import chat, debug_retrieve, health
from app.api.schemas import ChatRequest, RetrieveRequest
from app.scripts.test_answer_quality import TEST_QUERIES
from app.services.analytics_answer import generate_answer
from app.ui.backend import call_api


class FreeDemoTests(unittest.TestCase):
    def test_representative_queries_without_network_or_model(self):
        with patch.dict(os.environ, {"ANSWER_MODE": "analytics", "INSIGHT_SEARCH": "keyword"}), \
             patch("requests.sessions.Session.request", side_effect=AssertionError("network access")):
            for query, _ in TEST_QUERIES:
                with self.subTest(query=query):
                    answer = chat(ChatRequest(query=query))
                    self.assertIn("Source: bundled historical IPL data", answer.answer)
                    self.assertNotIn("LLM SKIPPED", answer.answer)
                    self.assertGreater(answer.structured_count + answer.insight_count, 0)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_leaderboard_facts_match_retrieval(self):
        query = "Who are the top IPL run scorers?"
        facts = debug_retrieve(RetrieveRequest(query=query))
        answer = chat(ChatRequest(query=query)).answer
        for row in facts.structured_results:
            self.assertIn(row["data"]["batter"], answer)
            self.assertIn(str(int(row["data"]["runs"])), answer)

    def test_definitions_and_unknowns(self):
        self.assertIn("per 100 balls", chat(ChatRequest(query="What does strike rate mean?")).answer)
        self.assertIn("couldn't find", chat(ChatRequest(query="zzzzzzzzzz")).answer)
        self.assertNotIn("Captains should", generate_answer("matchup", "Interpretation Hints:\nCaptains should target this matchup.", "matchup"))

    def test_powerplay_comparison_does_not_claim_career_totals_are_phase_stats(self):
        query = "Is rohit sharma a better batter to Virat Kohli in powerplay?"
        answer = chat(ChatRequest(query=query)).answer
        self.assertIn("cannot determine who is better in that phase", answer)
        self.assertIn("overall career comparison only", answer)
        self.assertIn("V Kohli leads", answer)
        self.assertIn("RG Sharma leads", answer)
        self.assertIn("How to read this", answer)
        self.assertIn("percentage points", answer)

    def test_phase_entities_require_whole_words(self):
        from app.services.query_router import extract_entities
        self.assertEqual(extract_entities("Compare Kohli and Rohit")['phases'], [])
        self.assertEqual(extract_entities("Compare Kohli and Rohit in powerplay")['phases'], ['Powerplay'])
        self.assertEqual(extract_entities("powerplays")['phases'], [])

    def test_metric_question_leads_with_requested_metric(self):
        answer = chat(ChatRequest(query="Compare Rohit and Kohli boundary percentage")).answer
        self.assertTrue(answer.startswith("**Boundary-ball percentage: RG Sharma leads"))
        self.assertIn("17.68%", answer)
        self.assertIn("1.36 percentage points", answer)

    def test_semantic_failure_uses_keyword_search(self):
        from app.services.insight_retriever import retrieve_insights
        with patch.dict(os.environ, {"INSIGHT_SEARCH": "semantic"}), \
             patch("app.services.insight_retriever._get_model", side_effect=OSError("offline")):
            self.assertTrue(retrieve_insights("Kohli"))

    def test_standalone_ui_backend(self):
        with patch.dict(os.environ, {"API_URL": ""}):
            self.assertEqual(call_api("/health")["answer_mode"], "analytics")
            self.assertIn("8661", call_api("/chat", {"query": "Who are the top IPL run scorers?"})["answer"])
        self.assertFalse(health().model_loaded)


if __name__ == "__main__":
    unittest.main()
