from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

class GoogleSearchParams(BaseModel):
    search_query: str
    num_results: int = Field(default=2, description="The number of results to return from the Google search")

class StockDataParams(BaseModel):
    ticker: str = Field(description="The stock ticker symbol to get data for, e.g. 'AAPL'")
    date: str = Field(description="The date to get stock data for, in the format 'YYYY-MM-DD'")

class ComputeParams(BaseModel):
    wolfram_query: str = Field(description="The mathematical computation / operation to perform, e.g. 'integrate cos(x)/sqrt(x) from 0 to 1'")

class GetWeatherParams(BaseModel):
    location: str = Field(description="The location to get weather data for, in the format 'City, Country'")
    date: str = Field(description="The date to get weather data for, in the format 'YYYY-MM-DD'")
    hour: str = Field(default="12", description="The hour of the day to get weather data for, in the format 'HH' (24-hour format)")

class APIResponse(BaseModel):
    api_name: str = Field(description="The name of the API to use for the query. Must be one of: 'google_search', 'get_stock_data', 'compute', 'get_weather'")


class BaseTool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the tool with validated arguments."""
        pass