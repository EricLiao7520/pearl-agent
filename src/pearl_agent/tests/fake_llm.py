"""Scripted replacement for generate_together; never makes API calls.

    llm = FakeLLM(["SIMPLE", {"api_name": "web_search"}])
    router = ToolRouter(registry=registry, llm_fn=llm)
    agent = MultiLMAgent(router=router, llm_fn=llm)

Use a callback instead of queued replies when parallel calls need different
responses: FakeLLM(lambda request: ...). The callback receives model, messages,
temperature, response_format, max_tokens, and any additional keyword arguments.
Use fake tools too when testing workflows entirely offline.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
from threading import Lock
from typing import Any, Callable, Iterable


@dataclass
class FakeMessage:
    content: str


Reply = str | dict | list | Exception


class FakeLLM:
    """Return queued replies or call a handler, recording every request.

    Dictionaries/lists become JSON text, strings are returned unchanged, and
    exception objects are raised. Queue access is thread-safe, but concurrent
    call order is unspecified; use a handler for model/prompt-specific replies.
    """

    def __init__(
        self,
        responses: Iterable[Reply] | Callable[[dict[str, Any]], Reply],
    ):
        if isinstance(responses, (str, dict)):
            raise TypeError('Wrap replies in a list, for example FakeLLM(["hello"])')
        self.calls: list[dict[str, Any]] = []
        self._handler = responses if callable(responses) else None
        self._responses = deque() if self._handler else deque(responses)
        self._lock = Lock()

    def __call__(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> FakeMessage:
        request = deepcopy({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format,
            "max_tokens": max_tokens,
            **kwargs,
        })
        with self._lock:
            self.calls.append(deepcopy(request))
            if self._handler is None:
                if not self._responses:
                    raise RuntimeError("FakeLLM has no scripted replies left")
                reply = self._responses.popleft()

        if self._handler is not None:
            reply = self._handler(request)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (dict, list)):
            reply = json.dumps(reply)
        if not isinstance(reply, str):
            raise TypeError("FakeLLM replies must be strings, JSON objects/lists, or exceptions")
        return FakeMessage(content=reply)
