# main.py
import asyncio
import logging
import os
import datetime

import click
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route # Ensure Route is imported if needed for specific routes()
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from adk_agent import create_agent
from adk_agent_executor import ADKAgentExecutor
from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


load_dotenv()

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10007)
def main(host: str, port: int):
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv(
        'GOOGLE_API_KEY'
    ):
        raise ValueError(
            'GOOGLE_API_KEY environment variable not set and '
            'GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
        )

    skill = AgentSkill(
        id='CalculatorSkill',
        name='Can Perform Calculations',
        description="Can provide calculation results for addition, subtraction, multiplication, and division.",
        tags=['calculator'],
        examples=['what is the addition of 5 and 10?', 'what is the multiplication of 2 and 3?'],
    )

    agent_card = AgentCard(
        name='Calculator Agent (Simulated)',
        description="An agent that can provide calculation results for addition, subtraction, multiplication, and division.",
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    adk_agent = asyncio.run(create_agent())
    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    agent_executor = ADKAgentExecutor(runner, agent_card)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    # Instantiate A2AStarletteApplication
    a2a_app_instance = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    # --- THE NEW ATTEMPT: Initialize Starlette with the A2A app's routes ---
    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ]

    # This is the most direct way to get the routes if A2AStarletteApplication
    # itself isn't directly callable as an ASGI app.
    # THIS ASSUMES `a2a_app_instance` HAS A `.routes` PROPERTY OR `.routes()` METHOD
    # THAT RETURNS A LIST OF STARLETTE ROUTES.
    # If a2a_app_instance.routes is a property:
    # app = Starlette(debug=True, routes=a2a_app_instance.routes, middleware=middleware)
    # If a2a_app_instance.routes() is a method:
    app = Starlette(debug=True, routes=a2a_app_instance.routes(), middleware=middleware)


    # Run the base Starlette application
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()