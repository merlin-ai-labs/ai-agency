#!/bin/bash
echo "🧪 Testing Weather Agent"
echo ""
echo "Test 1: Weather query"
curl -s -X POST http://localhost:8000/weather-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Paris?", "tenant_id": "test"}' | python3 -m json.tool
echo ""
echo ""
echo "Test 2: General chat (no tool needed)"
curl -s -X POST http://localhost:8000/weather-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "tenant_id": "test"}' | python3 -m json.tool
