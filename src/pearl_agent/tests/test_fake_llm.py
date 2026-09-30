"""Offline functional checks: python -m unittest pearl_agent.tests.test_fake_llm."""

import json
import unittest

from pearl_agent.core.agent import MultiLMAgent
from pearl_agent.core.research import DeepResearchAgent
from pearl_agent.core.router import ToolRouter
from pearl_agent.tests.fake_llm import FakeLLM
from pearl_agent.tools.base import BaseTool, WebSearchParams
from pearl_agent.tools.registry import ToolRegistry


class FakeSearchTool(BaseTool):
    name = "web_search"
    description = "Return a fixed offline search result."
    args_schema = WebSearchParams

    def run(self, search_query, num_results=2, **kwargs):
        return [{
            "title": "Test source",
            "link": "https://example.test/source",
            "snippet": "The answer is 42.",
            "webpage_content": "The answer is 42.",
        }][:num_results]


def make_agent(llm):
    registry = ToolRegistry()
    registry.register(FakeSearchTool())
    router = ToolRouter(registry=registry, llm_fn=llm)
    return MultiLMAgent(router=router, llm_fn=llm)


class FakeLLMTests(unittest.TestCase):
    def test_replies_requests_and_errors(self):
        llm = FakeLLM(["hello", {"choice": 0}, ValueError("simulated failure")])
        messages = [{"role": "user", "content": "question"}]
        self.assertEqual(llm(model="fake", messages=messages, temperature=0).content, "hello")
        messages[0]["content"] = "changed"
        self.assertEqual(llm.calls[0]["messages"][0]["content"], "question")
        self.assertEqual(llm.calls[0]["temperature"], 0)
        self.assertEqual(json.loads(llm(model="fake", messages=[]).content), {"choice": 0})
        with self.assertRaisesRegex(ValueError, "simulated failure"):
            llm(model="fake", messages=[])
        with self.assertRaisesRegex(RuntimeError, "no scripted replies"):
            llm(model="fake", messages=[])

    def test_single_tool_pipeline(self):
        llm = FakeLLM([
            "SIMPLE",
            {"api_name": "web_search"},
            {"search_query": "answer", "num_results": 1},
            "The answer is 42.",
        ])
        self.assertEqual(make_agent(llm).run_pipeline("What is the answer?"), "The answer is 42.")
        self.assertEqual(len(llm.calls), 4)
        schema = llm.calls[1]["response_format"]["schema"]
        self.assertEqual(schema["properties"]["api_name"]["enum"], ["web_search"])
        self.assertIn("The answer is 42", llm.calls[-1]["messages"][0]["content"])

    def test_invalid_tool_selection(self):
        agent = make_agent(FakeLLM([{"api_name": "unknown"}]))
        result = agent.router.route_query("question")
        self.assertEqual(result["error"], "Invalid API selected: unknown")

    def test_iterative_refinement(self):
        llm = FakeLLM([
            "<sub-query>Find the answer</sub-query>",
            {"api_name": "web_search"},
            {"search_query": "answer"},
            "<final_answer>The answer is 42.</final_answer>",
        ])
        self.assertEqual(make_agent(llm).iterative_refine("question"), "The answer is 42.")
        self.assertIn("The answer is 42", llm.calls[-1]["messages"][0]["content"])

    def test_parallel_fusion(self):
        def handler(request):
            prompt = request["messages"][0]["content"]
            if "expert task planner" in prompt:
                return "<sub-query>First question</sub-query><sub-query>Second question</sub-query>"
            if "You are an API router" in prompt:
                return {"api_name": "web_search"}
            if "extracts function parameters" in prompt:
                return {"search_query": request["messages"][-1]["content"]}
            if "expert judge and aggregator" in prompt:
                return "Fused answer: 42."
            if "helpful synthesis assistant" in prompt:
                return "The answer is 42."
            raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

        llm = FakeLLM(handler)
        self.assertEqual(make_agent(llm).decompose_and_fuse("question"), "Fused answer: 42.")
        self.assertEqual(len(llm.calls), 9)

    def test_research(self):
        llm = FakeLLM([
            "<sub-query>Find the answer</sub-query>",
            "The answer is 42. [Test source](https://example.test/source)",
        ])
        agent = make_agent(llm)
        research = DeepResearchAgent(tool_registry=agent.router.registry, llm_fn=llm)
        result = research.research("question")
        self.assertEqual(result["sources"], ["https://example.test/source"])
        self.assertIn("The answer is 42", result["report"])
        self.assertIn("https://example.test/source", llm.calls[-1]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
