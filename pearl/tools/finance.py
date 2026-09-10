import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pearl.tools.base import BaseTool, StockDataParams

class StockDataTool(BaseTool):
    name = "get_stock_data"
    description = "Retrieves historical stock data (open, high, low, close, volume) for a given ticker and YYYY-MM-DD date."
    args_schema = StockDataParams
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")

    def run(self, ticker: str, date: str, **kwargs: Any) -> Dict[str, Any]:
            """Retrieves historical stock data (open, high, low, close, volume) for a given ticker (e.g. 'AAPL') and YYYY-MM-DD date (e.g. '2025-03-05')."""
            
            # This function should return a dictionary containing the date, open, high, low, close, and volume of the stock data.
            # If there is an error, return a dictionary with an "error" key containing the error message.
            ################ CODE STARTS HERE ################
            try:
                url = f"https://api.massive.com/v1/open-close/{ticker.upper()}/{date}?adjusted=true&apiKey={self.api_key}"
                res = requests.get(url, timeout=10)
                data = res.json()
                if res.status_code == 200 and data.get("status") == "OK":
                    return {
                        "date": data.get("from", date),
                        "open": data.get("open"),
                        "high": data.get("high"),
                        "low": data.get("low"),
                        "close": data.get("close"),
                        "volume": data.get("volume")
                    }
                return {"error": data.get("message", f"Unable to fetch stock data for {ticker} on {date}")}
            except Exception as e:
                return {"error": str(e)}