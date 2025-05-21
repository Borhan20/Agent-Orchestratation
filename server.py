# server.py
import os
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel
import uvicorn
import datetime
import pytz # For timezone-aware datetime

# Define a Pydantic model for tool arguments if needed
class AddNumbersRequest(BaseModel):
    a: float
    b: float

class TimezoneRequest(BaseModel):
    timezone: str

app = FastAPI(
    title="Remote MCP Tools Server",
    description="Exposes simple tools for OpenAI Agents via MCP."
)

# --- Define your tools as FastAPI endpoints ---

@app.get("/add", operation_id="add_numbers")
async def add_numbers(request: AddNumbersRequest):
    """
    Adds two numbers together.
    :param a: The first number.
    :param b: The second number.
    """
    result = request.a + request.b
    return {"result": result, "description": f"The sum of {request.a} and {request.b} is {result}."}

@app.get("/get_current_time", operation_id="get_current_time")
async def get_current_time():
    """
    Returns the current UTC time.
    """
    now_utc = datetime.datetime.now(pytz.utc)
    return {"current_time_utc": now_utc.isoformat(), "description": f"Current UTC time is {now_utc.isoformat()}."}

@app.get("/get_time_in_timezone", operation_id="get_time_in_timezone")
async def get_time_in_timezone(request: TimezoneRequest):
    """
    Returns the current time in a specified IANA timezone (e.g., 'America/New_York', 'Europe/London').
    :param timezone: The IANA timezone string.
    """
    try:
        tz = pytz.timezone(request.timezone)
        now_in_tz = datetime.datetime.now(tz)
        return {"current_time": now_in_tz.isoformat(), "timezone": request.timezone, "description": f"Current time in {request.timezone} is {now_in_tz.isoformat()}."}
    except pytz.exceptions.UnknownTimeZoneError:
        return {"error": "Unknown timezone", "description": f"The timezone '{request.timezone}' is not recognized. Please provide a valid IANA timezone string."}

# --- Initialize and mount FastApiMCP ---
# This will automatically expose the FastAPI endpoints as MCP tools.
# The `base_url` is crucial for the client to correctly construct tool call URLs.
# For local development, use http://127.0.0.1:8000
mcp = FastApiMCP(
    app,
    name="MyRemoteTools",
    description="A collection of remote tools for AI agents.",
    # base_url="http://127.0.0.1:8000" # IMPORTANT: Change this to your public URL if deploying remotely
)
mcp.mount()

if __name__ == "__main__":
    # To run this server:
    # 1. Make sure you have uvicorn installed: pip install uvicorn
    # 2. Run from your terminal: uvicorn server:app --reload --port 8000
    # The --reload flag is useful for development as it restarts the server on code changes.
    print("Starting FastAPI MCP Server...")
    print("Access the API docs at: http://127.0.0.1:8000/docs")
    print("The MCP endpoint will be at: http://127.0.0.1:8000/mcp")
    uvicorn.run(app, host="127.0.0.1", port=8000)

