import datetime
import asyncio
from typing import Any, Dict, List
import logging

from google.adk.agents import LlmAgent # type: ignore[import-untyped]
# We no longer need to import 'Tool' or 'types' for simple function tools.
# from google.adk.tools import Tool
# from google.genai import types

# Define a simple function tool that simulates calendar access
async def get_statement_profit_loss( # Renamed for clarity, removed leading underscore
    a: float,
) -> Dict[str, Any]:
    
    await asyncio.sleep(0.5)
    logging.info(f"Simulated marketing request received: a={a}")

    if a <0 :
        return "loss"
    elif a > 0:
        return "profit"
    else:
        return "neutral"

    

async def create_agent() -> LlmAgent:
    """Constructs the ADK agent with a simulated calendar tool."""

    return LlmAgent(
        model='gemini-2.0-flash-001', # Or a similar capable model
        name='marketing_agent',
        description="An agent that give decision on profit or loss.",
        instruction=f"""
You are an agent that can decide whether the statement is profit or loss based on given inputs.
before getting statement of profit or loss you first need to calculate the profit or loss based on the given inputs.
user: will provide the sell price and consume price.
Today is {datetime.datetime.now()}.
""",
        # Simply pass the Python function directly to the 'tools' argument.
        # ADK will infer the tool schema from the function's signature and docstring.
        tools=[get_statement_profit_loss]
    )