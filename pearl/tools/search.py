import os
from typing import Dict, List, Any, Optional
from tavily import TavilyClient
from pearl.tools.base import BaseTool, WebSearchParams

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Performs a Tavily search for general knowledge questions, recent events, or information not covered by other tools."
    args_schema = WebSearchParams

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if self.api_key:
            self.client = TavilyClient(api_key=self.api_key)
        else:
            self.client = None

    def run(self, search_query: str, num_results: int = 2, **kwargs: Any) -> List[Dict[str, Any]]:
        """Performs a Google search for general knowledge questions, recent events, or information not covered by other tools."""
        if not self.client:
            return [{"error": "Tavily API Key not configured."}]
        try:
            response = self.client.search(query=search_query, max_results=num_results, include_raw_content=True)
            results = []
            for item in response.get("results", []):
                raw = item.get("raw_content", "") or ""
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "webpage_content": raw[:2000]
                })
            return results
        except Exception as e:
            # Error Case B: API failure, authentication error, or connection exception
            return [{"error": str(e)}]