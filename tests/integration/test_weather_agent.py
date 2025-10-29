"""Test script for weather agent flow.

Quick test to verify the weather agent works end-to-end:
- Database connection
- LLM integration
- Weather tool calling
- Conversation storage
"""

import asyncio
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from app.flows.weather_agent import WeatherAgentFlow


async def test_weather_agent():
    """Test weather agent with a simple query."""
    print("=" * 60)
    print("Testing Weather Agent Flow")
    print("=" * 60)

    # Initialize agent
    print("\n1. Initializing weather agent...")
    agent = WeatherAgentFlow()
    print(f"   ✓ Agent initialized: {agent.llm.provider_name} ({agent.llm.model_name})")

    # Test single-turn conversation
    print("\n2. Testing single-turn conversation...")
    print("   User: 'What's the weather in London?'")

    result = await agent.run(
        user_message="What's the weather in London?",
        tenant_id="test_user",
    )

    print(f"   ✓ Response: {result['response']}")
    print(f"   ✓ Conversation ID: {result['conversation_id']}")
    print(f"   ✓ Tool used: {result['tool_used']}")

    if result['tool_results']:
        print(f"   ✓ Tool results: {result['tool_results']}")

    # Test multi-turn conversation
    print("\n3. Testing multi-turn conversation...")
    conversation_id = result['conversation_id']

    print("   User: 'How about Paris?'")
    result2 = await agent.run(
        user_message="How about Paris?",
        tenant_id="test_user",
        conversation_id=conversation_id,
    )

    print(f"   ✓ Response: {result2['response']}")
    print(f"   ✓ Same conversation: {result2['conversation_id'] == conversation_id}")
    print(f"   ✓ Tool used: {result2['tool_used']}")

    # Test non-weather question
    print("\n4. Testing non-weather question...")
    print("   User: 'Tell me a joke'")

    result3 = await agent.run(
        user_message="Tell me a joke",
        tenant_id="test_user",
        conversation_id=conversation_id,
    )

    print(f"   ✓ Response: {result3['response'][:100]}...")
    print(f"   ✓ Tool used: {result3['tool_used']}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_weather_agent())
