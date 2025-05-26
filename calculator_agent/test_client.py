# client.py
import logging
import asyncio
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (AgentCard, MessageSendParams, SendMessageRequest,
                       SendStreamingMessageRequest)


async def main() -> None:
    PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
    EXTENDED_AGENT_CARD_PATH = "/agent/authenticatedExtendedCard"

    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- IMPORTANT: Ensure this matches your server's host and port ---
    base_url = 'http://localhost:10007' # Changed from 9999 to 10007 as per your server's default

    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(f"Attempting to fetch public agent card from: {base_url}{PUBLIC_AGENT_CARD_PATH}")
            _public_card = await resolver.get_agent_card()
            logger.info("Successfully fetched public agent card:")
            logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
            final_agent_card_to_use = _public_card
            logger.info("\nUsing PUBLIC agent card for client initialization (default).")

            if _public_card.supportsAuthenticatedExtendedCard:
                try:
                    logger.info(f"\nPublic card supports authenticated extended card. Attempting to fetch from: {base_url}{EXTENDED_AGENT_CARD_PATH}")
                    auth_headers_dict = {"Authorization": "Bearer dummy-token-for-extended-card"}
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={"headers": auth_headers_dict}
                    )
                    logger.info("Successfully fetched authenticated extended agent card:")
                    logger.info(_extended_card.model_dump_json(indent=2, exclude_none=True))
                    final_agent_card_to_use = _extended_card
                    logger.info("\nUsing AUTHENTICATED EXTENDED agent card for client initialization.")
                except Exception as e_extended:
                    logger.warning(f"Failed to fetch extended agent card: {e_extended}. Will proceed with public card.", exc_info=True)
            elif _public_card:
                logger.info("\nPublic card does not indicate support for an extended card. Using public card.")

        except Exception as e:
            logger.error(f"Critical error fetching public agent card: {e}", exc_info=True)
            raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e

        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
        logger.info("A2AClient initialized.")

        
        # --- Interactive Loop for Messages ---
        while True:
            user_input = input("Enter your message (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break

            logger.info(f"\n--- Sending message: '{user_input}' ---")

            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': user_input}, # Use user_input here
                    ],
                    'messageId': uuid4().hex,
                },
            }
            request = SendMessageRequest(
                params=MessageSendParams(**send_message_payload)
            )

        
            # For streaming responses, use send_message_streaming
            # For a regular message, send_message waits for the final response
            response = await client.send_message(request)
            print("Regular message response:")
            final_response = response.model_dump(mode='json', exclude_none=True)
            logging.info(f'Response from send_message: {final_response['result']['artifacts'][0]['parts'][0]['text']}')
            # print(response.model_dump(mode='json', exclude_none=True))

            # # --- Example 2: Streaming message ---
            # logger.info("\n--- Sending a streaming message ---")
            # streaming_request = SendStreamingMessageRequest(
            #     params=MessageSendParams(**send_message_payload)
            # )

            # print("Streaming message chunks:")
            # stream_response = client.send_message_streaming(streaming_request)
            # async for chunk in stream_response:
            #     print(chunk.model_dump(mode='json', exclude_none=True))


if __name__ == '__main__':
    asyncio.run(main())