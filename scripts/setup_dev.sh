#!/bin/bash
set -e

echo "=========================================="
echo "AI Agency Platform - Development Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ "$PYTHON_VERSION" < "$REQUIRED_VERSION" ]]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python version OK: $PYTHON_VERSION${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created!${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo ""

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -e .
pip install -e ".[dev]"
echo -e "${GREEN}Dependencies installed!${NC}"
echo ""

# Check if Docker is running
echo -e "${YELLOW}Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}Docker is running!${NC}"
echo ""

# Start PostgreSQL with docker-compose
echo -e "${YELLOW}Starting PostgreSQL with docker-compose...${NC}"
docker-compose up -d db
echo ""

# Wait for PostgreSQL to be healthy
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while ! docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}Error: PostgreSQL failed to start after $MAX_RETRIES attempts${NC}"
        docker-compose logs db
        exit 1
    fi
    echo "Waiting for PostgreSQL... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo -e "${GREEN}PostgreSQL is ready!${NC}"
echo ""

# Set DATABASE_URL for migrations
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ai_agency"

# Run database migrations
echo -e "${YELLOW}Running database migrations...${NC}"
alembic upgrade head
echo -e "${GREEN}Migrations completed!${NC}"
echo ""

# Install pre-commit hooks
echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
pre-commit install
echo -e "${GREEN}Pre-commit hooks installed!${NC}"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}.env file created! Please update with your actual values.${NC}"
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi
echo ""

# Success message
echo -e "${GREEN}=========================================="
echo "Development environment setup complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Run 'source venv/bin/activate' to activate the virtual environment"
echo "3. Run 'uvicorn app.main:app --reload' to start the development server"
echo "4. Or run 'docker-compose up' to start all services"
echo ""
echo "Useful commands:"
echo "  - Run tests: pytest"
echo "  - Run linting: ruff check app tests"
echo "  - Run formatting: ruff format app tests"
echo "  - Run type checking: mypy app"
echo "  - Create migration: alembic revision --autogenerate -m 'description'"
echo "  - Apply migrations: alembic upgrade head"
echo ""
echo -e "${GREEN}Happy coding!${NC}"
