from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import json, re
from pearl_agent.core.llm import generate_together
from pearl_agent.verification.datastructures import JudgeConfig

class LLMJudge:
    """
    LLM-as-a-judge that selects the index of the best code snippet from a list of candidate completions.
    If no candidate seems correct, the LLM returns None.
    """

    def __init__(self, llm_fn: Callable[..., Any] = generate_together, cfg: JudgeConfig = JudgeConfig(), model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.llm = llm_fn
        self.cfg = cfg
        self.model_name = model_name

    def _build_messages(self, problem_prompt: str, function_name: str, code_snippets: List[str]) -> str:
        candidates_formatted = []
        for i, candidate in enumerate(code_snippets):
            candidates_formatted.append(f"Candidate {i}:\n```python\n{candidate}\n```")
        candidates_str = "\n".join(candidates_formatted)
        max_idx = len(code_snippets) - 1
        prompt = f"""You are an expert Python code evaluator and judge.
Your task is to carefully evaluate the candidate implementations for the function `{function_name}` and choose the single best and most correct one.

Target Function:
{function_name}

Problem Statement and Requirements:
{problem_prompt}

Candidate Implementations:
{candidates_str}

Instructions:
1. Analyze each candidate carefully against all constraints, edge cases, and requirements described in the problem.
2. Select the index (from 0 to {max_idx}) of the candidate that is fully correct and bug-free.
3. If multiple candidates are correct, choose the one with the cleanest and most idiomatic implementation.
4. If none of the candidates correctly solve the problem, set "choice" to null.

Output Requirements:
You MUST respond with ONLY a single-line valid JSON object matching the schema below. Do NOT output any markdown code blocks, backticks, conversational preamble, or postscript.

Schema:
{{"choice": <integer index or null>, "reason": "<one short sentence>"}}

Examples:
{{"choice": 0, "reason": "Correctly handles all edge cases and boundary conditions."}}
{{"choice": null, "reason": "None of the candidates satisfy the negative digit sum requirements."}}"""
        return prompt

    def _parse_json_choice(self, raw: str) -> Tuple[Optional[int], str]:
        """
        Robustly extracts the judge's choice and reason from the raw LLM text response.

        Args:
            raw (str): The raw text output from the LLM judge.

        Returns:
            A tuple containing:
                - Optional[int]: The chosen index (or None if unparseable, None-chosen, or invalid)
                - str: The LLM's reasoning for the choice
        """
        if not raw or not raw.strip():
            return None, "Empty response"

        first = raw.strip().splitlines()[0].strip()
        obj = None
        try:
            obj = json.loads(first)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    obj = None

        if not isinstance(obj, dict):
            return None, "Unparseable"

        choice = obj.get("choice", None)
        reason = obj.get("reason", "")
        if choice is None:
            return None, reason or "None"

        try:
            idx = int(choice)
            return (idx if idx >= 0 else None), reason
        except Exception:
            return None, reason or "Non-integer index"
            
    def judge(
        self,
        problem_prompt: str,
        function_name: str,
        code_snippets: List[str],
    ) -> Dict[str, Any]:
        """
        Orchestrates the judging process by building a prompt, querying the LLM, and parsing the response.
        Args:
            problem_prompt (str): The problem specification.
            function_name (str): The name of the target function.
            code_snippets (List[str]): A list of candidate code snippets.

        Returns:
            A dictionary containing:
            - "choice": the int index of the chosen code snippet, or None
            - "reason": the stripped string that describes why the model chose the option
            - "raw_response": the raw response from the LLM for debugging
        """
        assert 1 <= len(code_snippets) <= self.cfg.max_choices

        prompt = self._build_messages(problem_prompt, function_name, code_snippets)
        response = self.llm(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
        )
        raw_response = response.content
        if isinstance(raw_response, list):
            raw_response = raw_response[0]
        final_choice, reason =  self._parse_json_choice(raw_response)
        return {"choice": final_choice, "reason": reason, "raw_response": raw_response}
        