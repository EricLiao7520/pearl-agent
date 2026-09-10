import os
from typing import Dict, Any, Optional
import sympy as sp
import nest_asyncio
import wolfram_api
from pearl.tools.base import BaseTool, ComputeParams

class MathTool(BaseTool):
    name = "compute"
    description = "Performs mathematical computations, calculus, and equation solving using SymPy and Wolfram Alpha."
    args_schema = ComputeParams

    def __init__(self, wolfram_app_id: Optional[str] = None):
        self.wolfram_app_id = wolfram_app_id or os.getenv("WOLFRAM_APP_ID")
        self.wolfram_client = None
        if self.wolfram_app_id:
            self.wolfram_client = wolfram_api.Client(self.wolfram_app_id)
        else:
            raise ValueError("wolfram_app_id key is not imported")
        
    def run(self, wolfram_query: str, **kwargs: Any) -> Dict[str, str]:
        """Performs a mathematical computation or operation using Wolfram Alpha."""
        if not self.wolfram_client:
            return {"error": "Wolfram client is not initialized."}
        try:
            nest_asyncio.apply()
            res = self.wolfram_client.query(wolfram_query)
            result_text = None
            try:
                result_text = next(res.results).text
            except (StopIteration, AttributeError):
                pass
            if not result_text:
                return {"error": f"No computation result found for query: {wolfram_query}"}
            return {"result": result_text}
        except Exception as e:
            return {"error": str(e)}