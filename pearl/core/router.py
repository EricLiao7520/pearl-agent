import json
from typing import Dict, Any, Optional, Callable
from pearl.tools.base import APIResponse
from pearl.tools.registry import ToolRegistry

class ToolRouter:
    def __init__(
        self, 
        registry: ToolRegistry, 
        llm_fn: Callable[..., Any], 
        router_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    ):
        self.registry = registry
        self.llm = llm_fn
        self.router_model = router_model

    def _parse_query_params(
        self,
        query: str, 
        tool_name: str,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Uses an LLM to parse parameters for a given function from a user query.

        Args:
            query (str): The natural language user query.
            function_name (str): The name of the API function to parse parameters for.
            model (Optional[str]): The specific model to use for parsing. Defaults to the
                instance's router_model.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of validated parameters, or None if
            parsing or validation fails.
        """
        tool = self.registry.get(tool_name)
        if not tool or not self.llm:
            return None
        target_model = model or self.router_model
        schema = tool.args_schema.model_json_schema()

        system_prompt = (
            f"You are a helpful assistant that extracts function parameters from natural language queries.\n"
            f"Function name: {tool.name}\n"
            f"Function description: {tool.description}\n"
            f"Extract the arguments conforming strictly to this JSON Schema:\n{json.dumps(schema)}\n"
            f"Respond ONLY with a valid JSON object matching the schema."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract parameters for the following query: {query}"}
        ]
        try:
            response = self.llm(
                model=target_model,
                messages=messages,
                response_format={"type": "json_object", "schema": schema}
            )
            content = response.content 
            parsed_json = json.loads(content)
            validated = tool.args_schema(**parsed_json)
            return validated.model_dump()

        except Exception:
            return None
        
    def route_query(self, query: str, model: Optional[str] = None) -> Dict:
        """Determines the appropriate API for a query, parses parameters, and executes the API call.

        This method first uses an LLM to decide which API tool is best suited for the
        user's query. It then calls `parse_query_params` to extract the necessary
        arguments and finally executes the chosen API function.

        Args:
            query (str): The natural language user query.
            model (Optional[str]): The specific model to use for routing. Defaults to the
                instance's router_model.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the API call, the
            name of the API used, the parameters, and any potential errors.
        """
        target_model = model or self.router_model
        tools_desc = self.registry.get_descriptions()
        system_prompt = (
            "You are an API router. Select the single best API tool for the user query.\n"
            "Available tools:\n{tools_desc}\n\n"
            f"You must respond with a JSON object conforming strictly to this schema:\n{json.dumps(APIResponse.model_json_schema())}"
        )
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Determine the appropriate API tool for this query: {query}"}
                ]
        try:
            response = self.llm(
                model=target_model,
                messages=messages,
                response_format={"type": "json_object", "schema": APIResponse.model_json_schema()}
            )
            content = response.content
            parsed = json.loads(content)
            validated = APIResponse(**parsed)
            api_name = validated.api_name
        except Exception as e:
            return {
                "api_used": None,
                "params": None,
                "results": None,
                "error": f"Failed to route query: {str(e)}"
            }
        tool = self.registry.get(api_name)
        if not tool:
            return {
                "api_used": api_name,
                "params": None,
                "results": None,
                "error": f"Invalid API selected: {api_name}"
            }
        params = self._parse_query_params(query, api_name, target_model)
        if params is None:
            return {
                "api_used": api_name,
                "params": None,
                "results": None,
                "error": f"Failed to parse parameters for {api_name}"
            }
        try:
            api_results = tool.run(**params)
            error = None
            #For other three APIs
            if isinstance(api_results, dict) and "error" in api_results:
                error = api_results["error"]
            #For google_search
            elif isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], dict) and "error" in api_results[0]:
                error = api_results[0]["error"]
            return {
                "api_used": api_name,
                "params": params,
                "results": api_results,
                "error": error
            }
        except Exception as e: 
            return {
                "api_used": api_name,
                "params": params,
                "results": None,
                "error": str(e)
            }