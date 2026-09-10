# examples/run_agent.py
import os
from dotenv import load_dotenv
from pearl.tools.registry import ToolRegistry
from pearl.core.router import ToolRouter
from pearl.core.agent import MultiLMAgent
from pearl.core.research import DeepResearchAgent
from pearl.core.llm import generate_together

load_dotenv()

def main():
    print("Initializing Pearl Agent Framework...")
    
    # 1. Initialize Tool Registry with default tools
    registry = ToolRegistry.default()
    
    # 2. Initialize Tool Router with generate_together as the LLM backend
    router = ToolRouter(
        registry=registry,
        llm_fn=generate_together,
        router_model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    )
    
    # 3. Initialize Multi-LM Agent (Decomposition, Fusion, Iterative Refinement)
    agent = MultiLMAgent(router=router)
    
    # Test Simple / Single Tool Query
    query_simple = "What is the stock closing price for TSLA on 2025-08-27?"
    print(f"\n[Test 1] Running pipeline on: '{query_simple}'")
    result_simple = agent.run_pipeline(query_simple)
    print(f"Result:\n{result_simple}")
    
    # 4. Initialize Deep Research Agent
    research_agent = DeepResearchAgent(tool_registry=registry)
    
    query_research = "What was the macroeconomic performance of the UK in 2024?"
    print(f"\n[Test 2] Running Deep Research on: '{query_research}'")
    research_output = research_agent.research(query_research)
    
    print("\n--- Final Research Report ---")
    print(research_output["report"])
    print("\n--- Sources Cited ---")
    for src in research_output["sources"]:
        print(f"- {src}")

if __name__ == "__main__":
    main()