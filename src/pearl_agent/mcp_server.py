import os
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from typing import Dict, Any, List

#Tools
from pearl_agent.tools.search import WebSearchTool
from pearl_agent.tools.finance import StockDataTool
from pearl_agent.tools.math import MathTool
from pearl_agent.tools.weather import WeatherTool

#Pydantic Schema
from pearl_agent.tools.base import (
    WebSearchParams,
    StockDataParams,
    ComputeParams,
    GetWeatherParams
)

load_dotenv()

mcp = MCPServer("pearl-agent")

web_tool = WebSearchTool()
stock_tool = StockDataTool()
math_tool = MathTool()
weather_tool = WeatherTool()

@mcp.tool()
def web_search(params: WebSearchParams) -> List[Dict[str, Any]]:
    """Performs a Tavily search for general knowledge questions, recent events, or information not covered by other tools."""
    return web_tool.run(search_query=params.search_query, num_results=params.num_results)

@mcp.tool()
def get_stock_data(params: StockDataParams) -> Dict[str, Any]:
    """Retrieves historical stock data (open, high, low, close, volume) for a given ticker and YYYY-MM-DD date."""
    return stock_tool.run(ticker=params.ticker, date=params.date)

@mcp.tool()
def compute(params: ComputeParams) -> Dict[str, str]:
    """Performs mathematical computations, calculus, and equation solving using SymPy and Wolfram Alpha."""
    return math_tool.run(wolfram_query=params.wolfram_query)

@mcp.tool()
def get_weather(params: GetWeatherParams) -> Dict[str, Any]:
    """Fetches weather data (temperature, humidity, wind speed) for a specific location, YYYY-MM-DD date, and HH hour."""
    return weather_tool.run(location=params.location, date=params.date, hour=params.hour)

if __name__ == "__main__":
    mcp.run(transport="stdio")