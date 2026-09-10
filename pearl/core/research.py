from typing import Dict, Any, List
import re
from pearl.tools.registry import ToolRegistry
from pearl.core.llm import generate_together

class DeepResearchAgent:
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
        self.generation_temp = 0.7
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def generate(
            self, 
            query: str, 
            model: str = "wayliao7520-fbe8/llama",
            temperature: float = 0.7
        ) -> str:
            """Generates a response to the query using the specified model."""
            message = [{"role": "user", "content": query}]
            try:
                response = generate_together(model=model, messages=message, temperature=temperature)
                return response.content
            except Exception as e:
                return f"Error generating response: {e}"

    def _decompose_research_query(self, query: str, max_sub_queries: int = 4) -> List[str]:
        prompt= (
            f"You are an expert research planner. Decompose the following complex research topic into up to {max_sub_queries} distinct, highly specific search queries."
            "Guidelines: \n"
            "1. Each sub-query must be fully self-contained (include specific entities, dates, symbols, or formulas).\n"
            "2. Enclose EACH sub-query within <sub-query>...</sub-query> tags.\n\n"
            f"User query: {query}"
        )
        response = self.generate(prompt)
        parsed_results = re.findall(r"<sub-query>(.*?)</sub-query>", response)
        parsed_results = [q.strip() for q in parsed_results]
        if not parsed_results:
            print("Warning: No subqueries are generated, fallback to orignal query")
            return [query.strip()]
        return parsed_results[:max_sub_queries]

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conducts deep research on a given query.
        
        Args:
            query: Complex research question to investigate
            
        Returns:
            Dictionary containing:
            - report: str, synthesized findings with citations
            - sources: List[str], list of source URLs or references
        """
        sub_queries = self._decompose_research_query(query, max_sub_queries=3)
        sources: List[str] = []
        collected_evidence: List[str] = []

        search_tool = self.registry.get("google_search")
        for sub_q in sub_queries:
            results = search_tool.run(sub_q, num_results=2)
            if not results:
                continue

            for item in results:
                if isinstance(item, dict) and "error" not in item:
                    link = item.get("link")
                    title = item.get("title", "Online Source")
                    snippet = item.get("snippet", "")
                    content = item.get("webpage_content", "")

                    if link and link not in sources:
                        sources.append(link)

                    evidence_block = (
                        f"Source Title: {title}\n"
                        f"URL: {link}\n"
                        f"Summary: {snippet}\n"
                        f"Content: {content}"
                    )
                    collected_evidence.append(evidence_block)

        context_str = "\n\n---\n\n".join(collected_evidence) if collected_evidence else "No search results available."
        system_instruction = (
            "You are an elite research analyst. Using ONLY the provided search findings, "
            "write an extensive, cohesive, and deeply insightful research report addressing the user's question.\n\n"
            "Report Requirements:\n"
            "1. Length & Structure: Write exactly 4 to 5 substantial paragraphs organized logically "
            "(Executive overview, specific developments/policies, measured impacts/milestones, and future outlook/challenges).\n"
            "2. Chronological Precision: Clearly track temporal data (specifically highlighting 2021-2024 milestones).\n"
            "3. In-Text Citations: Directly cite findings inline using markdown links pointing to the sources "
            "(e.g., [Source Title](URL)). Every factual paragraph should include citations.\n"
            "4. Objective Tone: Professional, authoritative, and fact-focused without filler words."
        )

        synthesis_prompt = (
            f"{system_instruction}\n\n"
            f"Research Question:\n{query}\n\n"
            f"Gathered Evidence:\n{context_str}\n\n"
            "Final Research Report:"
        )
        report = self.generate(
            query=synthesis_prompt,
            model=self.model,
            temperature=self.generation_temp
        )

        return {
            "report": report.strip(),
            "sources": sources
        }