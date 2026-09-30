import os
import time
from typing import List, Dict, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Global client to be initialized once
_client = None

def get_client() -> OpenAI:
    """Initializes OpenAI client configured for Together AI or OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY")
    base_url = (
        "https://api.together.xyz/v1"
        if (os.getenv("TOGETHER_API_KEY") and not os.getenv("OPENAI_API_KEY"))
        else None
    )
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

def generate_together(
    model: str,
    messages: List[Dict],
    temperature: Optional[float] = None,
    response_format: Optional[Dict] = None,
    max_tokens: Optional[int] = None
):
    """
    Generates a response from a model on Together AI with support for structured output.
    """
    client = get_client()

    args = {
        "model": model,
        "messages": messages,
    }
    #Optional parameter for GPT-4 and other models
    if temperature is not None:
        args["temperature"] = temperature
    
    if response_format:
        args["response_format"] = response_format
    if max_tokens:
        args["max_tokens"] = max_tokens
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**args)
            return response.choices[0].message
        except Exception as e:
            print(f"API call failed on attempt {attempt + 1}: {e}")
            print(f"\tArguments: {args}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Final attempt failed. Returning None.")
                return None

class OpenAISampler:
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        n_samples: int = 1,
        max_tokens: int = 4096,
        max_workers: int = 16,
    ):
        self.client = get_client()
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
    def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """
        Makes a chat completion request with retry logic.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.

        Returns:
            str: The response from the model.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Allows the instance to be called as a function to send prompts.

        Args:
            prompts (str or List[str]): A prompt or a list of prompts to send to the model.

        Returns:
            str or List[str]: The response(s) from the model.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        expanded_prompts = [p for p in prompts for _ in range(self.n_samples)]
        responses = [None] * len(expanded_prompts)