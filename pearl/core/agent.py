from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import json
import re

from pearl.tools.registry import ToolRegistry
from pearl.core.router import ToolRouter
from pearl.core.llm import generate_together

class MultiLMAgent:
    """A class to manage multiple language models for generation, iterative refinement, and fusion"""
    def __init__(
        self,
        router: ToolRouter,
        decomposition_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        iterative_refinement_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        fusion_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        generation_temp: float = 0.7,
    ):
        self.router = router
        self.decomposition_model = decomposition_model
        self.iterative_refinement_model = iterative_refinement_model
        self.fusion_model = fusion_model
        self.generation_temp = generation_temp

    def generate(
        self,
        query: str, 
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.7
    ):
        message = [{"role": "user", "content": query}]
        try:
            response = generate_together(model=model, messages=message, temperature=temperature)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"

    def single_LM_with_single_API_call(self, query: str, model: str) -> str:
        """Generates a response by querying the API manager for the necessary data and then using an LM to generate a final response based on the API output."""
        api_choice = self.router.route_query(query)
        api_used = api_choice.get("api_used")
        results = api_choice.get("results")
        error = api_choice.get("error")
        system_instructions = (
            "You are a helpful assistant. Answer the user's query using the provided API tool output contexts. \n"
            "If the tool returned an error or no relevant data, using your internal knowledge to give the best possible answer. \n"
            "State the final answer directly and clearly."
        )
        context_str = f"API used: {api_used}\nAPI results: {json.dumps(results) if results else None}\n"
        if error:
            context_str += f"API Errors: {error}"
        prompt = (
            f"{system_instructions}\n\n"
            f"Context from External Tool: {context_str}\n\n"
            f"User Query: {query}"
        )
        return self.generate(query=prompt, model=model, temperature=self.generation_temp)

    def _get_query_decomposition_prompt(self, query: str) -> str:
        """Helper function to generate a prompt for the LLM to decompose the query into sub-queries."""
        prompt= (
            "You are an expert task planner and query decomposer. Your goal is to break down a complex, multihop user query."
            "into smaller, self-contained sub-queries that can be answered independently by external tools.\n"
            "Guidelines: \n"
            "1. Each sub-query must be fully self-contained (include specific entities, dates, symbols, or formulas).\n"
            "2. Enclose EACH sub-query within <sub-query>...</sub-query> tags.\n\n"
            f"User query: {query}"
        )
        return prompt

    def _get_sub_queries(self, query: str, max_sub_queries: int = 4) -> List[str]:
        """Helper function to break down a complex user query into smaller subqueries."""
        prompt = self._get_query_decomposition_prompt(query)
        response = self.generate(prompt,model=self.decomposition_model, temperature=self.generation_temp)
        parsed_results = re.findall(r"<sub-query>(.*?)</sub-query>", response)
        parsed_results = [q.strip() for q in parsed_results]
        if not parsed_results:
            print("Warning: No subqueries are generated, fallback to orignal query")
            return [query.strip()]
        return parsed_results[:max_sub_queries]

    def decompose_query(self, query: str, max_sub_queries: int = 4) -> List[Dict[str, Any]]:
        """Decomposes a query into independent, more manageable sub-queries that are executed in parallel via the API manager.

        Args:
            query (str): The user's original query.
            max_sub_queries (int): The maximum number of sub-queries to generate.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each contains the
            sub-query, the API used to answer the sub-query, parameters, results, and any errors.
        """
        def process_subquery(sub_q: str) -> Dict[str, Any]:
            res = self.router.route_query(sub_q)
            res["sub_query"] = sub_q
            return res
        subqueries = self._get_sub_queries(query, max_sub_queries=max_sub_queries)
        with ThreadPoolExecutor(max_workers=max_sub_queries) as exe:
            result = exe.map(process_subquery, subqueries)
        return list(result)

    def _get_synthesis_prompt(self, query: str, decomposed_queries: List[Dict[str, Any]] = None) -> str:
        """
        Constructs a prompt for an LLM to synthesize a final response to the user's query, using the API results of the decomposed sub-queries.
        """
        system_instruction = (
            "You are an helpful synthesis assistant. Your task is to provide an accurate, coherent,"
            "and direct answer to the user's original query by synthesizing the evidence gathered "
            "from multiple sub-query tool executions.\n\n"
            "Guidelines:\n"
            "1. Ground your response strictly in the provided API results and facts.\n"
            "2. If an API call resulted in an error or returned no data, rely on the remaining valid evidence "
            "or your core knowledge, but do not hallucinate specific figures or dates.\n"
            "3. Answer the user's question directly and concisely without exposing raw JSON, code snippets, "
            "or internal tool mechanics unless necessary."
        )
        evidence_blocks = []
        if decomposed_queries:
            for idx, decomposed_query in enumerate(decomposed_queries):
                sub_query = decomposed_query.get("sub_query")
                api_used = decomposed_query.get("api_used")
                results = decomposed_query.get("results")
                error = decomposed_query.get("error")
                block = f"Sub-Query {idx}: {sub_query}\nAPI: {api_used}\nResults: {json.dumps(results) if results else 'None'}"
                if error:
                    block += f"\nError: {error}"
                evidence_blocks.append(block)
        evidence_str = "\n\n".join(evidence_blocks) if evidence_blocks else "No tool evidence available."    
        prompt = (
            f"{system_instruction}\n\n"
            f"Gathered Evidence:\n{evidence_str}\n\n"
            f"User Query:\n{query}\n\n"
        )
        return prompt

    def decompose_and_fuse(self, query: str) -> str:
        """
        Decomposes a user query into independent sub-queries for API execution, then synthesizes the resulting outputs into a single, coherent response by prompting multiple models and fusing their generated answers.
        """
        
        candidate_models = [
            "google/gemma-4-E4B-it",
            "meta-llama/Llama-3.1-8B-Instruct",
            "OpenAI/gpt-oss-20B"
        ]
        decomposed_queries = self.decompose_query(query)
        synthesis_prompt = self._get_synthesis_prompt(query=query, decomposed_queries=decomposed_queries)
        def sample_model(model: str) -> str:
            response = self.generate(query=synthesis_prompt, model=model, temperature=self.generation_temp)
            return response
        with ThreadPoolExecutor(max_workers=len(candidate_models)) as exe:
            result = exe.map(sample_model, candidate_models)
        model_result = list(result)
        valid_responses = [
            res for res in model_result 
            if res and not res.startswith("Error generating response")
        ]
        if len(valid_responses) == 1:
            return valid_responses[0]
        if not valid_responses:
            return self.generate(synthesis_prompt, model=self.fusion_model, temperature=self.generation_temp)
        formatted_responses = "\n\n".join(
            f"Candidate {idx}:\n{res.strip()}" 
            for idx, res in enumerate(valid_responses, 1)
        )
        fuse_prompt = (
            "You are an expert judge and aggregator. Below is the user's original query, along with candidate responses generated by different language models.\n\n"
            "Your task:\n"
            "1. Analyze the candidate responses and identify the factual consensus.\n"
            "2. Resolve any contradictions or discrepancies by favoring verified factual details.\n"
            "3. Synthesize the responses into a single, direct, coherent, and comprehensive final answer.\n"
            "4. Do not include commentary comparing the models. State the final answer directly.\n\n"
            f"User Query:\n{query}\n\n"
            f"Candidate Responses:\n{formatted_responses}"
        )
        fuse_response = self.generate(fuse_prompt, model=self.fusion_model, temperature=self.generation_temp)
        return fuse_response

    def _get_iterative_refinement_prompt(self, original_query: str, history: List[Dict[str, Any]]) -> str:
        """
        Helper function to generate the prompt for the iterative refinement model.
        """
        system_instruction = (
            "You are an iterative reasoning assistant solving a complex task step-by-step. "
            "At each step, examine the original query and the history of actions executed so far.\n\n"
            "You must choose EXACTLY ONE of two actions:\n"
            "1. If you need more information or if a previous tool call failed, generate the next natural language sub-query.\n"
            " Wrap your sub-query strictly in: <sub-query>your sub-query here</sub-query>\n"
            "2. If the history contains sufficient information to fully answer the original query, synthesize the final answer.\n"
            "Wrap your final answer strictly in: <final_answer>your final synthesized answer here</final_answer>\n\n"
            "Rules:\n"
            "- Do not output both tags. Choose only one action.\n"
            "- Keep sub-queries specific, self-contained, and focused on missing facts."
        )
        history_blocks = []
        history_str = ""
        if history:
            for idx, step in enumerate(history, 1):
                sub_q = step.get("sub_query", "Unknown")
                api_used = step.get("api_used", "None")
                results = step.get("results")
                error = step.get("error")
                block = (
                    f"Step {idx}: Sub-query: {sub_q}, API Used: {api_used}, Result: {json.dumps(results) if results is not None else 'None'}"
                )
                if error:
                    block += f" Error: {error}"
                history_blocks.append(block)
            history_str = "\n\n".join(history_blocks)
        prompt = (
            f"{system_instruction}\n\n"
            f"Original Query:\n{original_query}\n\n"
            f"Execution History:\n{history_str}\n\n"
            f"Next Action:"
        )
        return prompt

    def iterative_refine(self, query: str, max_iterations: int = 3) -> str:
        """
        Sequentially generates and executes natural language sub-queries to answer a complex query. At each step, the model can either issue a new sub-query (e.g. if additional information is needed, or if a prior sub-query failed), or return the final answer if it has enough information.
        """
        history = []
        for i in range(max_iterations):
            prompt = self._get_iterative_refinement_prompt(original_query=query, history=history)
            response = self.generate(
                query=prompt, 
                model=self.iterative_refinement_model, 
                temperature=self.generation_temp
            )
            final_answer_match = re.search(r"<final_answer>(.*?)</final_answer>", response, re.DOTALL)
            if final_answer_match:
                return final_answer_match.group(1).strip()
            sub_match = re.search(r"<sub-query>(.*?)</sub-query>", response, re.DOTALL)
            if sub_match:
                sub_q = sub_match.group(1).strip()
                res = self.router.route_query(sub_q)
                res["sub_query"] = sub_q
                history.append(res)
            else:
                # If tags were omitted, treat text as response
                return response.strip()
        #fallback
        synthesis_prompt = self._get_synthesis_prompt(query=query, decomposed_queries=history)
        return self.generate(query=synthesis_prompt, model=self.iterative_refinement_model, temperature=0.2)

    def run_pipeline(self, query: str) -> str:
        """
        Runs the full agentic pipeline for a given query.
        """
        router_prompt = (
        "Classify the following query into exactly one category:\n"
        "- PARALLEL: Needs independent facts collected simultaneously\n"
        "- SEQUENTIAL: Multi-step where later steps depend on earlier answers\n"
        "- SIMPLE: Can be answered with a single tool lookup\n\n"
        f"Query: {query}\n"
        "Category:"
        )
        decision = self.generate(router_prompt, model=self.fusion_model, temperature=0.0).strip()

        if "PARALLEL" in decision:
            return self.decompose_and_fuse(query)
        elif "SIMPLE" in decision:
            return self.single_LM_with_single_API_call(query, model=self.decomposition_model)
        else:
            return self.iterative_refine(query, max_iterations=4)