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
