# Pearl Agent

Pearl Agent is a Python framework for answering questions with language models and external tools. It routes a query to an appropriate tool, gathers evidence, and can use several models to plan, refine, and synthesize an answer.

## Features

- **Natural-language tool routing** — the router supplies registered tool names and descriptions to an LLM, extracts schema-validated arguments, and returns the tool result and any error.
- **Built-in tools** — web search via Tavily, historical daily stock data, mathematical computation with Wolfram Alpha, and historical/hourly weather lookup.
- **Multi-model answering** — choose a single-lookup flow, parallel query decomposition with answer fusion, or sequential iterative refinement.
- **Deep research** — split a topic into focused searches, collect result snippets and page content, and produce a report with a list of source URLs.
- **Agent evaluation** — compare answering strategies on math, web search, weather, and stock questions using simple answer matching or an optional LLM judge.
- **Offline functional tests** — inject a scripted fake LLM and fake tools to check routing, refinement, fusion, and research without API calls.
- **MCP server** — expose web search, stock data, math, and weather tools over MCP stdio transport.
- **Code evaluation utilities** — run candidate Python code in a constrained Docker container, verify it against HumanEval-style or structured test cases, and use an LLM judge to select among candidates.

## Installation

Requires Python 3.12 or newer and uv.

```bash
git clone <repository-url>
cd pearl-agent
uv sync
```

`uv sync` creates the project environment and installs dependencies from `pyproject.toml` and the committed `uv.lock` file. Run Python commands in that environment with `uv run`, for example:

```bash
uv run python -m pearl_agent.mcp_server
```

## Configuration

Copy the example environment file, then add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and fill in the values for the services you plan to use. The example includes `TOGETHER_API_KEY`, `TAVILY_API_KEY`, `POLYGON_API_KEY`, `WOLFRAM_APP_ID`, and `HUGGINGFACE_HUB_TOKEN`.

`OPENAI_API_KEY` takes precedence when both model keys are set. Without it, `TOGETHER_API_KEY` configures the OpenAI-compatible Together endpoint. Tool credentials are needed only for the corresponding services. The default registry constructs the Wolfram-backed math tool, so `WOLFRAM_APP_ID` must be set when using `ToolRegistry.default()`.

## Quick start

The example initializes the default tools and router, then runs a query through the agent pipeline:

```python
from dotenv import load_dotenv

from pearl_agent.core.agent import MultiLMAgent
from pearl_agent.core.llm import generate_together
from pearl_agent.core.router import ToolRouter
from pearl_agent.tools.registry import ToolRegistry

load_dotenv()

registry = ToolRegistry.default()
router = ToolRouter(registry=registry, llm_fn=generate_together)
agent = MultiLMAgent(router=router, llm_fn=generate_together)

answer = agent.run_pipeline("What was the closing price of TSLA on 2025-08-27?")
print(answer)
```

The pipeline classifies the request and selects a simple lookup, parallel decomposition and fusion, or iterative refinement flow. You can also call a specific flow directly:

```python
answer = agent.single_LM_with_single_API_call(
    "What was the closing price of TSLA on 2025-08-27?",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

answer = agent.decompose_and_fuse(
    "Compare the weather and major economic indicators in London and Paris last year."
)

answer = agent.iterative_refine(
    "How has the price of AAPL changed since 2025-01-02?",
    max_iterations=3,
)
```

For a research report and its source URLs:

```python
from pearl_agent.core.research import DeepResearchAgent

researcher = DeepResearchAgent(tool_registry=registry, llm_fn=generate_together)
result = researcher.research("What was the macroeconomic performance of the UK in 2024?")
print(result["report"])
print(*result["sources"], sep="\n")
```

The router, agent, and research agent each accept `llm_fn`, a callable that returns an object with a `.content` attribute. Inject the same callable into each component to use a shared backend. The default helper includes `temperature` only when it is not `None`; individual workflows can still supply their own temperature settings.

## Agent evaluation

Run a small evaluation with live model calls:

```bash
uv run python -m pearl_agent.tests.evaluate_agent --strategy direct --max-rows 2
```

Compare all strategies, or select Tavily-backed web search questions:

```bash
uv run python -m pearl_agent.tests.evaluate_agent --strategy all --debug
uv run python -m pearl_agent.tests.evaluate_agent --strategy single_tool --question-type web_search --max-rows 2
```

- `--strategy`: `direct`, `single_tool`, `fusion`, `iterative`, `automatic`, or `all`.
- `--question-type`: `math`, `web_search`, `weather`, `stock`, or `all`.
- `--debug`: use two questions per category; `--max-rows` caps the total.
- `--model`: select the model for direct/single-tool answers and tool routing. Fusion and refinement retain the agent's configured models.
- `--judge-model`: optionally grade with an LLM; otherwise use simple answer matching.

The runner prints each answer, its reference answer, correctness, and accuracy per strategy. It preserves the existing historical question set and basic matching logic, so scores are a lightweight comparison rather than a comprehensive quality measure. Direct evaluation only needs model credentials; tool strategies also use the default tool registry and its configuration described above.

## Offline testing with a fake LLM

Run the functional tests without credentials or API calls:

```bash
uv run python -m unittest pearl_agent.tests.test_fake_llm -v
```

`FakeLLM` returns scripted replies and records requests in `llm.calls`. Strings become response text, dictionaries/lists become JSON text, and exception objects simulate failures. For parallel workflows, pass a callback that chooses a reply from the request's model or messages instead of relying on queue order.

To check the evaluation loop offline:

```python
from pearl_agent.tests.fake_llm import FakeLLM
from pearl_agent.tests.evaluate_agent import prepare_dataset, evaluate_strategy

dataset = prepare_dataset(question_type="math", max_rows=2)
llm = FakeLLM(["13", "999"])  # First answer correct, second incorrect.

def answer_fn(query):
    return llm(
        model="fake",
        messages=[{"role": "user", "content": query}],
    ).content

result = evaluate_strategy(dataset, answer_fn, judge_model=None)
assert result["accuracy"] == 0.5
assert len(llm.calls) == 2
```

For agent workflows, supply `llm_fn=llm` to both the router and agent (and research agent when used). Use a registry of fake tools as well: replacing the LLM alone does not prevent real tool API calls. `tests/test_fake_llm.py` includes a fake search tool and examples for the main workflows. These tests verify control flow and data passing, not a real model's answer quality.

## MCP server

Run the bundled MCP server with stdio transport:

```bash
uv run python -m pearl_agent.mcp_server
```

It registers `web_search`, `get_stock_data`, `compute`, and `get_weather` tools. Set the relevant service credentials before starting it.

## Code verification utilities

The `pearl_agent.execution` and `pearl_agent.verification` modules provide a Docker-based Python runner, a `CodeVerifier` for test snippets or structured `TestCase` inputs, and an `LLMJudge` for comparing candidate implementations. The runner expects Docker and a locally available `humaneval-sandbox` image by default. These utilities are library components; they are separate from the question-answering pipeline.

## Project layout

```text
src/pearl_agent/
├── core/          # LLM client, routing, agent workflows, and research
├── tools/         # Tool interfaces, registry, and built-in integrations
├── execution/     # Sandboxed code execution and test harness
├── verification/  # Test data structures, metrics, and LLM judge
├── tests/         # Live agent evaluation, fake LLM, and offline functional tests
└── mcp_server.py  # MCP stdio server
```

## Notes

- External tool calls depend on third-party APIs and their credentials. Results can be unavailable when a service rejects a request or has no data for the requested date.
- Weather lookup uses geocoding and Open-Meteo data; stock lookup uses the Massive/Polygon open-close endpoint.
- Model names in examples are provider model identifiers and may need to be changed to match your account's available models.
