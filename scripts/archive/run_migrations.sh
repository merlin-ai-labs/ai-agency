#!/bin/bash
set -e

echo "=========================================="
echo "Running Database Migrations"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    # Try to load from .env file
    if [ -f ".env" ]; then
        echo -e "${YELLOW}Loading DATABASE_URL from .env file...${NC}"
        export $(grep -v '^#' .env | grep DATABASE_URL | xargs)
    fi

    # Check again
    if [ -z "$DATABASE_URL" ]; then
        echo -e "${RED}Error: DATABASE_URL is not set${NC}"
        echo "Please set DATABASE_URL environment variable or add it to .env file"
        echo ""
        echo "Examples:"
        echo "  export DATABASE_URL=postgresql://user:pass@localhost:5433/ai_agency"
        echo "  export DATABASE_URL=postgresql://user:pass@/cloudsql/project:region:instance/db"
        exit 1
    fi
fi

echo -e "${GREEN}Using DATABASE_URL: ${DATABASE_URL//:*@/:***@}${NC}"
echo ""

# Parse command line arguments
COMMAND=${1:-upgrade}
REVISION=${2:-head}

case $COMMAND in
    upgrade)
        echo -e "${YELLOW}Upgrading database to revision: $REVISION${NC}"
        alembic upgrade $REVISION
        echo -e "${GREEN}Migration upgrade completed!${NC}"
        ;;

    downgrade)
        echo -e "${YELLOW}Downgrading database to revision: $REVISION${NC}"
        alembic downgrade $REVISION
        echo -e "${GREEN}Migration downgrade completed!${NC}"
        ;;

    current)
        echo -e "${YELLOW}Checking current database revision...${NC}"
        alembic current
        ;;

    history)
        echo -e "${YELLOW}Showing migration history...${NC}"
        alembic history
        ;;

    heads)
        echo -e "${YELLOW}Showing head revisions...${NC}"
        alembic heads
        ;;

    show)
        echo -e "${YELLOW}Showing migration details for: $REVISION${NC}"
        alembic show $REVISION
        ;;

    revision)
        MESSAGE=${2:-"Auto-generated migration"}
        echo -e "${YELLOW}Creating new migration: $MESSAGE${NC}"
        alembic revision --autogenerate -m "$MESSAGE"
        echo -e "${GREEN}Migration file created!${NC}"
        echo -e "${YELLOW}Review the generated file and run 'alembic upgrade head' to apply${NC}"
        ;;

    stamp)
        echo -e "${YELLOW}Stamping database with revision: $REVISION${NC}"
        alembic stamp $REVISION
        echo -e "${GREEN}Database stamped!${NC}"
        ;;

    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        echo "Usage: $0 [command] [revision]"
        echo ""
        echo "Commands:"
        echo "  upgrade [revision]    - Upgrade to a later version (default: head)"
        echo "  downgrade [revision]  - Downgrade to a previous version"
        echo "  current              - Show current revision"
        echo "  history              - Show revision history"
        echo "  heads                - Show head revisions"
        echo "  show [revision]      - Show details of a revision"
        echo "  revision [message]   - Create a new revision (autogenerate)"
        echo "  stamp [revision]     - Set database to a specific revision without running migrations"
        echo ""
        echo "Examples:"
        echo "  $0 upgrade head              # Upgrade to latest"
        echo "  $0 downgrade -1              # Downgrade one revision"
        echo "  $0 revision 'Add user table' # Create new migration"
        echo "  $0 current                   # Show current version"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
