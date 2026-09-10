import os
import requests
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from pearl.tools.base import BaseTool, GoogleSearchParams

class GoogleSearchTool(BaseTool):
    name = "google_search"
    description = "Performs a Google search for general knowledge questions, recent events, or information not covered by other tools."
    args_schema = GoogleSearchParams

    def __init__(self, api_key: Optional[str] = None, cx_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cx_id = cx_id or os.getenv("GOOGLE_CX_ID")

    def _extract_webpage_content(self, url: str) -> str:
        """Helper to scrape and clean text content from a URL.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            str: The extracted and cleaned text content, truncated to 2000 characters.
        """
        try:
            header = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"}
            response = requests.get(url,headers=header, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join(soup.stripped_strings)
            return text[:2000]
        except Exception:
            return ""

    def run(self, search_query: str, num_results: int = 2, **kwargs: Any) -> List[Dict[str, Any]]:
        """Performs a Google search for general knowledge questions, recent events, or information not covered by other tools."""
        if not self.api_key or not self.cx_id:
            return [{"error": "Google API Key or Search Engine CX ID not configured."}]
        try:
            service = build("customsearch", "v1", developerKey=self.api_key)
            res = service.cse().list(
                q=search_query,
                cx=self.cx_id,
                num=num_results
            ).execute()
            items = res.get("items", [])
            if not items:
                return [{"error": f"No search results found for query: {search_query}"}]
            
            results = []
            for item in items:
                link = item.get("link", "")
                results.append({
                    "title": item.get("title", ""),
                    "link": link,
                    "snippet": item.get("snippet", ""),
                    "webpage_content": self._extract_webpage_content(link)
                })
            return results
        except Exception as e:
            # Error Case B: API failure, authentication error, or connection exception
            return [{"error": str(e)}]