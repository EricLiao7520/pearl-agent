# Pearl Agent

Pearl Agent is a Python framework for answering questions with language models and external tools. It routes a query to an appropriate tool, gathers evidence, and can use several models to plan, refine, and synthesize an answer.

## Features

- **Natural-language tool routing** — an LLM selects a tool, extracts schema-validated arguments, and returns the tool result and any error.
- **Built-in tools** — web search via Tavily, historical daily stock data, mathematical computation with Wolfram Alpha, and historical/hourly weather lookup.
- **Multi-model answering** — choose a single-lookup flow, parallel query decomposition with answer fusion, or sequential iterative refinement.
- **Deep research** — split a topic into focused searches, collect result snippets and page content, and produce a report with a list of source URLs.
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
agent = MultiLMAgent(router=router)

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

researcher = DeepResearchAgent(tool_registry=registry)
result = researcher.research("What was the macroeconomic performance of the UK in 2024?")
print(result["report"])
print(*result["sources"], sep="\n")
```

## MCP server

Run the bundled MCP server with stdio transport:

```bash
python -m pearl_agent.mcp_server
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
├── examples/      # Example agent workflow
└── mcp_server.py  # MCP stdio server
```

## Notes

- External tool calls depend on third-party APIs and their credentials. Results can be unavailable when a service rejects a request or has no data for the requested date.
- Weather lookup uses geocoding and Open-Meteo data; stock lookup uses the Massive/Polygon open-close endpoint.
- Model names in examples are provider model identifiers and may need to be changed to match your account's available models.
