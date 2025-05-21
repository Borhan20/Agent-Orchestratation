# client.py
import asyncio
import os
from dotenv import load_dotenv
import openai
from agents import Agent, Runner
from agents.mcp import MCPServerSse # This is the client-side component for connecting to SSE MCP servers

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
# Ensure you have OPENAI_API_KEY set in your .env file or as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in a .env file or your system environment.")

# Define the URL of your remote MCP server
# This should match the `base_url` you configured in server.py
REMOTE_MCP_SERVER_URL = "http://127.0.0.1:8000/mcp" # The /mcp path is where FastApiMCP exposes its endpoint

async def main():
    print(f"Connecting to remote MCP server at: {REMOTE_MCP_SERVER_URL}")

    # Create an MCPServerSse instance. This acts as the client-side representation
    # of your remote MCP server.
    # The 'name' here is a label for your server within the Agent SDK.
    remote_mcp_server = MCPServerSse(
        params={"url": REMOTE_MCP_SERVER_URL},
        name="MyRemoteTools" # This name should match the 'name' given to FastApiMCP in server.py
    )

    # The 'async with' statement ensures the server connection is properly managed (opened and closed).
    async with remote_mcp_server:
        print("MCP Server connection established. Initializing agent...")

        # Initialize the Agent with instructions and the remote MCP server.
        # The agent will automatically discover the tools exposed by this server.
        agent = Agent(
            name="Tool-Using Assistant",
            instructions="You are a helpful assistant with access to remote tools. Use the 'add_numbers' tool to perform addition, 'get_current_time' to get the current UTC time, and 'get_time_in_timezone' to get the time in a specific timezone. Always try to use the tools to answer questions related to their functionality.",
            mcp_servers=[remote_mcp_server], # Pass the MCPServerSse instance here
        )

        print("\n--- Testing 'add_numbers' tool ---")
        prompt1 = "What is 123 plus 456?"
        print(f"User: {prompt1}")
        result1 = await Runner.run(starting_agent=agent, input=prompt1)
        print(f"Assistant: {result1.final_output}")

        print("\n--- Testing 'get_current_time' tool ---")
        prompt2 = "What is the current time?"
        print(f"User: {prompt2}")
        result2 = await Runner.run(starting_agent=agent, input=prompt2)
        print(f"Assistant: {result2.final_output}")

        print("\n--- Testing 'get_time_in_timezone' tool (valid timezone) ---")
        prompt3 = "What is the current time in Europe/London?"
        print(f"User: {prompt3}")
        result3 = await Runner.run(starting_agent=agent, input=prompt3)
        print(f"Assistant: {result3.final_output}")

        print("\n--- Testing 'get_time_in_timezone' tool (invalid timezone) ---")
        prompt4 = "What is the current time in Atlantis/LostCity?"
        print(f"User: {prompt4}")
        result4 = await Runner.run(starting_agent=agent, input=prompt4)
        print(f"Assistant: {result4.final_output}")

        print("\n--- Testing a general question (should not use tools) ---")
        prompt5 = "Tell me a fun fact about cats."
        print(f"User: {prompt5}")
        result5 = await Runner.run(starting_agent=agent, input=prompt5)
        print(f"Assistant: {result5.final_output}")

      
        while True:
            q = input("You> ").strip()
            if not q:
                break
            resp = await Runner.run(starting_agent=agent, input=q)
            # resp.final_output should be a MessageOutput
            print("Agent>", resp.final_output)


if __name__ == "__main__":
    # To run this client:
    # 1. Ensure your server.py is running (uvicorn server:app --reload --port 8000)
    # 2. Make sure you have your OPENAI_API_KEY set in a .env file or environment variable.
    # 3. Run from your terminal: python client.py
    asyncio.run(main())

