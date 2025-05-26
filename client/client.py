# ai_enhanced_client_simplified_fixed.py
import logging
import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum

import httpx
from openai import AsyncOpenAI

from a2a.client import A2ACardResolver, A2AClient # Assuming a2a library is installed
from a2a.types import (AgentCard, MessageSendParams, SendMessageRequest)

class QueryType(Enum):
    COMPUTATIONAL = "computational"
    MARKETING = "marketing"
    COLLABORATIVE = "collaborative"
    GENERAL = "general"


@dataclass
class AgentConfig:
    """Configuration for each A2A agent"""
    base_url: str
    name: str
    capabilities: List[str]
    description: str
    agent_type: str


class DynamicAgentOrchestrator:
    """AI-powered orchestrator that dynamically identifies agent needs"""

    def __init__(self, openai_api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.openai_client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        self.a2a_agents = {
            "calculator": AgentConfig(
                base_url='http://localhost:10007',
                name="Calculator Agent",
                capabilities=["mathematical_calculations", "numerical_analysis", "percentage_calculations",
                              "financial_computations", "statistical_analysis", "revenue_calculations",],
                description="Specialized in mathematical calculations, financial computations, and numerical analysis",
                agent_type="computational"
            ),
            "marketing": AgentConfig(
                base_url='http://localhost:10008',
                name="Marketing Agent",
                capabilities=["profit_loss_calculations", "marketing_strategy", "campaign_analysis", "market_research", "roi_analysis",
                              "brand_positioning", "customer_analysis", "competitive_analysis"],
                description="Expert in marketing strategies, campaign planning, and business analysis",
                agent_type="marketing"
            )
        }
        self.initialized_a2a_clients: Dict[str, A2AClient] = {}

    async def _invoke_additional_agent_tool(self, agent_response: str, original_query: str) -> str:
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the agent response to determine if additional agents are needed.
                        Available agents: calculator, marketing.
                        Look for indicators like:
                        - Mentions of needing calculations when response is from marketing agent
                        - Mentions of needing marketing analysis when response is from calculator
                        - Incomplete answers that require other expertise
                        - Requests for additional data that other agents can provide

                        Return JSON: {"needs_additional": true/false, "suggested_agent": "agent_key", "reason": "explanation"}
                        Example agent_key: "calculator" or "marketing"."""
                    },
                    {
                        "role": "user",
                        "content": f"Original query: {original_query}\nAgent response: {agent_response}"
                    }
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in invoke_additional_agent_tool: {e}")
            return json.dumps({"needs_additional": False, "reason": f"Analysis failed: {str(e)}"})

    async def initialize_a2a_client(self, agent_key: str) -> A2AClient:
        if agent_key in self.initialized_a2a_clients:
            return self.initialized_a2a_clients[agent_key]

        if agent_key not in self.a2a_agents:
            self.logger.error(f"Unknown agent key: {agent_key}")
            raise ValueError(f"Agent configuration not found for {agent_key}")
            
        config = self.a2a_agents[agent_key]
        try:
            timeout = httpx.Timeout(timeout=100.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as temp_http_client:
                resolver = A2ACardResolver(
                    httpx_client=temp_http_client,
                    base_url=config.base_url,
                )
                agent_card = await resolver.get_agent_card()
                self.logger.info(f"Fetched public A2A card for {config.name}")

                if agent_card.supportsAuthenticatedExtendedCard:
                    self.logger.info(f"Attempting to fetch authenticated extended card for {config.name}")
                    try:
                        auth_headers = {"Authorization": "Bearer dummy-token-for-extended-card"}
                        extended_card = await resolver.get_agent_card(
                            relative_card_path="/agent/authenticatedExtendedCard",
                            http_kwargs={"headers": auth_headers}
                        )
                        agent_card = extended_card
                        self.logger.info(f"Successfully fetched authenticated extended card for {config.name}")
                    except Exception as ext_e:
                        self.logger.warning(f"Failed to get authenticated extended card for {config.name}, using public card. Error: {ext_e}")
                        pass

                client = A2AClient(
                    httpx_client=httpx.AsyncClient(timeout=timeout),
                    agent_card=agent_card
                )
                self.initialized_a2a_clients[agent_key] = client
                self.logger.info(f"Initialized A2A client for {config.name}")
                return client
        except Exception as e:
            self.logger.error(f"Failed to initialize {config.name}: {e}")
            raise

    async def send_to_a2a_agent(self, agent_key: str, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        try:
            client = await self.initialize_a2a_client(agent_key)
            config = self.a2a_agents[agent_key]
            enhanced_message = message
            if context:
                enhanced_message = f"Previous Context: {context}\n\nCurrent Query: {message}"

            payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': enhanced_message}],
                    'messageId': uuid4().hex,
                },
            }
            request = SendMessageRequest(params=MessageSendParams(**payload))
            response = await client.send_message(request)
            result_dump = response.model_dump(mode='json', exclude_none=True)
            response_text = "No response text found."
            if result_dump.get('result', {}).get('artifacts'):
                 parts = result_dump['result']['artifacts'][0].get('parts')
                 if parts and parts[0].get('text'):
                     response_text = parts[0]['text']
            
            return {
                'agent': config.name,
                'agent_key': agent_key,
                'response': response_text,
                'success': True
            }
        except Exception as e:
            self.logger.error(f"Error with {agent_key}: {e}")
            # --- Fix for AttributeError starts here ---
            agent_config = self.a2a_agents.get(agent_key)
            agent_name = agent_config.name if agent_config else agent_key
            # --- Fix for AttributeError ends here ---
            return {
                'agent': agent_name,
                'agent_key': agent_key,
                'response': f"Error: {str(e)}", # Original error 'e' is preserved
                'success': False
            }

    async def analyze_query_with_ai(self, query: str) -> Dict[str, Any]:
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the user query and determine what agents are needed.
                        Available agents and their keys:
                        - calculator: Can provide calculation results for addition, subtraction, multiplication, and division, also revenue calculation.
                        - marketing: Can give decision on profit or loss based on given inputs, marketing strategy, ROI.

                        Examples:
                        - "what is the addition of 5 and 10?" -> needs calculator agent
                        - "is -20 profit or loss?" -> needs marketing agent
                        - "calculate profit margin for campaign with $10,000 spend and $15,000 revenue" -> needs both calculator and marketing
                        - "hello" -> needs no specific agent, suggest empty list [] for agents

                        IMPORTANT: Return EXACTLY this JSON format:
                        {
                            "suggested_agents": ["calculator"] or ["marketing"] or ["calculator", "marketing"] or [],
                            "query_type": "computational" or "marketing" or "collaborative" or "general",
                            "reasoning": "explanation of your decision"
                        }"""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            self.logger.info(f"Raw AI analysis response: {content}")
            try:
                parsed = json.loads(content)
                raw_suggested_agents = parsed.get("suggested_agents", [])
                valid_agents = []
                if isinstance(raw_suggested_agents, list):
                    for agent_key in raw_suggested_agents:
                        if agent_key in self.a2a_agents:
                            valid_agents.append(agent_key)
                        elif isinstance(agent_key, str): # Allow minor variations
                            if agent_key.lower().startswith("calc"): valid_agents.append("calculator")
                            elif agent_key.lower().startswith("market"): valid_agents.append("marketing")
                            else: self.logger.warning(f"AI suggested unknown agent: {agent_key}")
                valid_agents = list(dict.fromkeys(valid_agents)) # Remove duplicates
                return {
                    "suggested_agents": valid_agents, # No default to calculator here
                    "query_type": parsed.get("query_type", QueryType.GENERAL.value),
                    "reasoning": parsed.get("reasoning", "AI analysis completed"),
                    "execution_strategy": "sequential" if len(valid_agents) > 1 else "single"
                }
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse AI response as JSON, using keyword-based fallback.")
                query_lower = query.lower()
                suggested_agents = []
                calc_keywords = ['calculate', 'add', 'subtract', 'multiply', 'divide', '+', '-', '*', '/', 'sum', 'math']
                if any(keyword in query_lower for keyword in calc_keywords):
                    suggested_agents.append('calculator')
                marketing_keywords = ['profit', 'loss', 'marketing', 'campaign', 'strategy', 'roi', 'revenue', 'business']
                if any(keyword in query_lower for keyword in marketing_keywords):
                    suggested_agents.append('marketing')
                suggested_agents = list(dict.fromkeys(suggested_agents))
                return { # No default to calculator in fallback either
                    "query_type": QueryType.COLLABORATIVE.value if len(suggested_agents) > 1 else (QueryType.COMPUTATIONAL.value if "calculator" in suggested_agents else (QueryType.MARKETING.value if "marketing" in suggested_agents else QueryType.GENERAL.value)),
                    "suggested_agents": suggested_agents,
                    "execution_strategy": "sequential" if len(suggested_agents) > 1 else "single",
                    "reasoning": "Keyword-based fallback analysis"
                }
        except Exception as e:
            self.logger.error(f"Error in AI query analysis: {e}")
            return { # Default to no agents if analysis completely fails
                "query_type": QueryType.GENERAL.value,
                "suggested_agents": [],
                "execution_strategy": "single",
                "reasoning": f"AI Analysis failed: {str(e)}. No specific agent could be determined."
            }

    async def process_query_intelligently(self, query: str) -> str:
        self.logger.info(f"Processing query with AI orchestration: {query}")
        try:
            analysis_data = await self.analyze_query_with_ai(query)
            self.logger.info(f"AI Analysis: {analysis_data}")

            initial_suggested_agents = analysis_data.get("suggested_agents", [])
            if not initial_suggested_agents:
                # Handle cases where no specific agent is needed or identified
                reasoning = analysis_data.get("reasoning", "No specific agent was identified for your query.")
                # You could have a general LLM call here if desired, or just a polite message.
                if query.lower() in ["hello", "hi", "hello there", "hey"]:
                    return f"Hello! {reasoning} How can I help you today?"
                return f"I've analyzed your query: \"{reasoning}\". No specific tool seems directly applicable. How can I assist you further?"

            processed_agents_keys: set[str] = set()
            results: List[Dict[str, Any]] = []
            current_context = ""
            agents_to_run = list(initial_suggested_agents)

            if agents_to_run:
                primary_agent_key = agents_to_run.pop(0)
                self.logger.info(f"Executing primary agent: {primary_agent_key}")
                primary_result = await self.send_to_a2a_agent(primary_agent_key, query)
                results.append(primary_result)
                processed_agents_keys.add(primary_agent_key)

                if primary_result['success']:
                    current_context = f"Analysis from {primary_result['agent']}: {primary_result['response']}"
                    additional_check_response = await self._invoke_additional_agent_tool(
                        agent_response=primary_result['response'], original_query=query
                    )
                    try:
                        additional_data = json.loads(additional_check_response)
                        if additional_data.get("needs_additional"):
                            dynamically_suggested_agent = additional_data.get("suggested_agent")
                            if dynamically_suggested_agent and dynamically_suggested_agent in self.a2a_agents and \
                               dynamically_suggested_agent not in processed_agents_keys:
                                self.logger.info(f"AI dynamically determined need for agent: {dynamically_suggested_agent}.")
                                if dynamically_suggested_agent not in agents_to_run:
                                     agents_to_run.insert(0, dynamically_suggested_agent)
                                else:
                                    agents_to_run.remove(dynamically_suggested_agent)
                                    agents_to_run.insert(0, dynamically_suggested_agent)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not parse additional agent analysis: {additional_check_response}")
                else:
                    self.logger.warning(f"Primary agent {primary_agent_key} failed. Context will not include its response.")
            
            for agent_key in agents_to_run:
                if agent_key not in processed_agents_keys:
                    self.logger.info(f"Executing subsequent agent: {agent_key} with context.")
                    result = await self.send_to_a2a_agent(agent_key, query, context=current_context)
                    results.append(result)
                    processed_agents_keys.add(agent_key)
                    if result['success']:
                        current_context += f"\nAdditional analysis from {result['agent']}: {result['response']}"
            
            return self._combine_results(query, results, analysis_data)
        except Exception as e:
            self.logger.exception(f"Error in intelligent processing for query '{query}': {e}")
            return f"Sorry, I encountered an error processing your query: {str(e)}"

    def _combine_results(self, query: str, results: List[Dict], analysis: Dict) -> str:
        successful_results = [r for r in results if r.get('success')]
        if not successful_results:
            errors = [f"{r.get('agent', 'Unknown Agent')}: {r.get('response', 'No error detail')}" for r in results if not r.get('success')]
            error_details = "\n".join(errors) if errors else "No specific error details available."
            return f"All agents encountered errors processing your query.\nDetails:\n{error_details}"

        if len(successful_results) == 1 and not any(not r.get('success') for r in results): # only one agent ran and it was successful
            return successful_results[0]['response']

        combined = f"**Comprehensive AI-Orchestrated Response for: {query}**\n\n"
        combined += f"*Initial AI Analysis Reason: {analysis.get('reasoning', 'Multi-agent collaboration required')}*\n"
        combined += f"*Agents initially considered: {', '.join(analysis.get('suggested_agents',[])) or 'None'}*\n\n"

        for i, result in enumerate(successful_results, 1):
            combined += f"**Analysis from {result.get('agent', 'Unknown Agent')}:**\n{result.get('response', 'No response text.')}\n\n"
            if i < len(successful_results):
                combined += "---\n\n"
        
        if any(not r.get('success') for r in results):
            combined += "**Errors Encountered During Processing:**\n"
            for r in results:
                if not r.get('success'):
                    combined += f"- {r.get('agent', 'Unknown Agent')}: {r.get('response', 'No error detail.')}\n"
            combined += "\n"
        return combined.strip()

    async def cleanup(self):
        self.logger.info("Cleaning up A2A clients...")
        for agent_key, client in self.initialized_a2a_clients.items():
            if hasattr(client, 'httpx_client') and client.httpx_client:
                try:
                    await client.httpx_client.aclose()
                    self.logger.info(f"Closed httpx client for {agent_key}")
                except Exception as e:
                    self.logger.error(f"Error closing client for {agent_key}: {e}")
        self.initialized_a2a_clients.clear()

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("FATAL: OPENAI_API_KEY not found in environment variables.")
        print("🚨 CRITICAL: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key to enable AI-powered agent orchestration.")
        return

    orchestrator = DynamicAgentOrchestrator()
    print("\n🤖 Simplified AI-Powered Dynamic A2A Client Ready! (Fixed)")
    print("   Focuses on direct AI calls for orchestration.")
    print("\n📋 Available A2A agents by key: calculator, marketing")
    print("   Ensure A2A agent servers are running (e.g., http://localhost:10007 for calculator).")
    print("Type 'quit' to exit, 'help' for examples.\n")

    try:
        while True:
            user_input = input("💭 Your query: ").strip()
            if user_input.lower() == 'quit':
                logger.info("User requested to quit.")
                break
            elif user_input.lower() == 'help':
                print("\n🎯 Example queries:")
                print("- 'Calculate the profit margin for a marketing campaign with $10,000 spend and $15,000 revenue'")
                print("- 'What's the ROI of my digital marketing campaign?'")
                print("- 'hello there' (should be handled gracefully now)")
                print("\n🧠 The AI will attempt to determine which agents to use and coordinate between them!\n")
                continue
            elif not user_input:
                continue

            print(f"\n🔍 AI analyzing query: \"{user_input}\"")
            logger.info(f"Received query: {user_input}")
            try:
                response = await orchestrator.process_query_intelligently(user_input)
                print(f"\n✅ AI-Orchestrated Response:\n{response}\n")
                print("=" * 80)
            except Exception as e:
                logger.exception(f"Unhandled error processing query in main loop: {e}")
                print(f"\n❌ Critical Error: {str(e)}\n")
    except KeyboardInterrupt:
        logger.info("User interrupted with Ctrl+C.")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.exception(f"Unexpected error in main execution: {e}")
        print(f"\n💥 An unexpected error occurred: {e}")
    finally:
        logger.info("Initiating cleanup...")
        await orchestrator.cleanup()
        logger.info("Cleanup finished. Exiting.")

if __name__ == '__main__':
    asyncio.run(main())