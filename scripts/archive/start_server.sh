#!/bin/bash
# Start FastAPI server for development

set -e

echo "Starting AI Agency FastAPI server..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set in environment"
fi

# Start server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
