"""Weather agent flow with LLM tool calling and conversation memory.

This flow demonstrates the standard pattern for agent flows:
1. Load conversation history from PostgreSQL
2. Add user message to history
3. Call LLM with tool definitions
4. If LLM wants to call tool -> execute tool
5. Call LLM again with tool result
6. Save all messages to PostgreSQL
7. Return response
"""

import json
import logging
from typing import Any
from uuid import uuid4

from sqlmodel import Session

from app.adapters.llm_factory import get_llm_adapter
from app.core.base import BaseFlow
from app.core.decorators import log_execution, timeout
from app.db.base import get_session
from app.db.repositories.conversation_repository import ConversationRepository
from app.tools.weather.v1 import get_weather

logger = logging.getLogger(__name__)


# Weather tool definition for OpenAI function calling
WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g., 'London', 'Paris', 'Tokyo')",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial", "standard"],
                    "description": "Temperature units. metric=Celsius, imperial=Fahrenheit, standard=Kelvin",
                    "default": "metric",
                },
            },
            "required": ["location"],
        },
    },
}


class WeatherAgentFlow(BaseFlow):
    """Weather agent with LLM tool calling and conversation memory.

    This agent can have natural conversations about weather, using the
    get_weather tool when needed to fetch real-time weather data.

    Example:
        >>> flow = WeatherAgentFlow()
        >>> result = await flow.run(
        ...     user_message="What's the weather in London?",
        ...     tenant_id="user_123"
        ... )
        >>> print(result["response"])
        "It's currently 15°C and cloudy in London"
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize weather agent flow.

        Args:
            provider: LLM provider ("openai", "vertex", "mistral"). If None, uses settings
            model: Model name. If None, uses provider default
        """
        super().__init__(
            name="weather_agent",
            description="Weather assistant with tool calling and memory",
            version="1.0.0",
        )

        # Initialize LLM adapter (provider-agnostic)
        self.llm = get_llm_adapter(provider=provider, model=model)

        logger.info(
            f"Initialized WeatherAgentFlow with {self.llm.provider_name} ({self.llm.model_name})",
            extra={
                "provider": self.llm.provider_name,
                "model": self.llm.model_name,
            },
        )

    @log_execution
    @timeout(seconds=120.0)
    async def run(
        self,
        user_message: str,
        tenant_id: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process user message through weather agent.

        Flow:
        1. Load conversation history from PostgreSQL
        2. Add user message to history
        3. Call LLM with weather tool definition
        4. If LLM wants to call tool -> execute get_weather()
        5. Call LLM again with tool result
        6. Save all messages to PostgreSQL
        7. Return response

        Args:
            user_message: User's message/question
            tenant_id: Tenant identifier for multi-tenancy
            conversation_id: Optional conversation ID. If None, creates new conversation

        Returns:
            {
                "response": str,           # Assistant's response
                "conversation_id": str,    # Conversation UUID
                "tool_used": bool,         # Whether weather tool was called
                "tool_results": dict | None  # Tool results if tool was used
            }

        Example:
            >>> result = await flow.run(
            ...     user_message="What's the weather in Paris?",
            ...     tenant_id="user_123"
            ... )
            >>> print(result["response"])
        """
        with Session(get_session()) as session:
            repo = ConversationRepository(session)

            # Step 1: Create or load conversation
            if not conversation_id:
                conversation_id = repo.create_conversation(
                    tenant_id=tenant_id,
                    flow_type="weather",
                )
                logger.info(f"Created new weather conversation {conversation_id}")
            else:
                if not repo.conversation_exists(conversation_id):
                    logger.warning(f"Conversation {conversation_id} not found, creating new one")
                    conversation_id = repo.create_conversation(
                        tenant_id=tenant_id,
                        flow_type="weather",
                        conversation_id=conversation_id,
                    )

            # Step 2: Load conversation history
            history = repo.get_conversation_history(conversation_id)
            logger.info(f"Loaded {len(history)} messages from conversation {conversation_id}")

            # Step 3: Add user message to history
            repo.save_message(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                flow_type="weather",
                role="user",
                content=user_message,
            )

            # Build messages for LLM (add system prompt if first message)
            messages = []
            if not history:
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a helpful weather assistant. "
                        "You can check current weather for any location using the get_weather tool. "
                        "Be friendly and conversational. "
                        "Always include the temperature and conditions in your responses."
                    ),
                })

            messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            # Step 4: Call LLM with tool definition
            tool_used = False
            tool_results = None
            assistant_message = None

            try:
                # Call LLM with tool definitions using adapter interface
                llm_response = await self.llm.complete_with_metadata(
                    messages=messages,  # type: ignore
                    tools=[WEATHER_TOOL_DEFINITION],  # type: ignore
                    tool_choice="auto",
                    tenant_id=tenant_id,
                )

                # Convert to expected format
                response = {
                    "content": llm_response["content"],
                    "role": "assistant",
                }
                if llm_response.get("tool_calls"):
                    response["tool_calls"] = llm_response["tool_calls"]

                # Check if LLM wants to call a tool
                if response.get("tool_calls"):
                    tool_used = True
                    logger.info("LLM requested tool call")

                    # Step 5: Execute tool and get result
                    tool_call = response["tool_calls"][0]  # Handle first tool call
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    # Execute get_weather tool
                    weather_result = await get_weather(
                        location=tool_args["location"],
                        tenant_id=tenant_id,
                        units=tool_args.get("units", "metric"),
                    )

                    tool_results = weather_result

                    # Step 6: Save assistant message with tool call
                    repo.save_message(
                        conversation_id=conversation_id,
                        tenant_id=tenant_id,
                        flow_type="weather",
                        role="assistant",
                        content=response.get("content", ""),
                        tool_calls=[
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    )

                    # Add tool result to messages
                    messages.append({
                        "role": "assistant",
                        "content": response.get("content") or "",
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    })

                    # Add tool response message
                    tool_response_content = json.dumps(weather_result)
                    messages.append({
                        "role": "tool",
                        "content": tool_response_content,
                        "tool_call_id": tool_call["id"],
                    })

                    # Save tool result message
                    repo.save_message(
                        conversation_id=conversation_id,
                        tenant_id=tenant_id,
                        flow_type="weather",
                        role="tool",
                        content=tool_response_content,
                        message_metadata={"tool_call_id": tool_call["id"]},
                    )

                    # Step 7: Call LLM again with tool result using adapter interface
                    final_llm_response = await self.llm.complete_with_metadata(
                        messages=messages,  # type: ignore
                        tools=[WEATHER_TOOL_DEFINITION],  # type: ignore
                        tool_choice="auto",
                        tenant_id=tenant_id,
                    )

                    assistant_message = final_llm_response.get("content", "")

                else:
                    # No tool call needed, use LLM response directly
                    assistant_message = response.get("content", "")

                # Step 8: Save final assistant message
                repo.save_message(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    flow_type="weather",
                    role="assistant",
                    content=assistant_message,
                )

                logger.info(
                    f"Weather agent completed successfully",
                    extra={
                        "conversation_id": conversation_id,
                        "tool_used": tool_used,
                        "tenant_id": tenant_id,
                    },
                )

                return {
                    "response": assistant_message,
                    "conversation_id": conversation_id,
                    "tool_used": tool_used,
                    "tool_results": tool_results,
                }

            except Exception as e:
                logger.exception(
                    f"Weather agent failed: {str(e)}",
                    extra={
                        "conversation_id": conversation_id,
                        "error": str(e),
                    },
                )
                raise

    async def validate(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for the flow.

        Args:
            input_data: Input data with user_message and tenant_id

        Returns:
            True if input is valid

        Example:
            >>> await flow.validate({"user_message": "Hello", "tenant_id": "user_123"})
            True
        """
        required_fields = ["user_message", "tenant_id"]

        for field in required_fields:
            if field not in input_data:
                logger.warning(f"Missing required field: {field}")
                return False

            if not input_data[field]:
                logger.warning(f"Empty required field: {field}")
                return False

        return True
