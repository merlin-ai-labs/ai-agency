#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "===================================="
echo "Wave 1 Validation Script"
echo "===================================="
echo ""

ERRORS=0

# Function to print result
check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        ERRORS=$((ERRORS + 1))
    fi
}

# 1. Check Docker is running
echo "1. Checking Docker..."
if docker ps > /dev/null 2>&1; then
    check_result 0 "Docker daemon is running"
else
    check_result 1 "Docker daemon is not running"
fi

# 2. Check PostgreSQL container
echo ""
echo "2. Checking PostgreSQL container..."
if docker ps | grep -q ai_agency_postgres; then
    check_result 0 "PostgreSQL container is running"

    # Check health status
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' ai_agency_postgres 2>/dev/null || echo "unknown")
    if [ "$HEALTH" = "healthy" ]; then
        check_result 0 "PostgreSQL is healthy"
    else
        check_result 1 "PostgreSQL health status: $HEALTH"
    fi
else
    check_result 1 "PostgreSQL container is not running"
    echo -e "${YELLOW}  Run: docker-compose up -d db${NC}"
fi

# 3. Check database connection
echo ""
echo "3. Checking database connection..."
if docker-compose exec -T db psql -U postgres -d ai_agency -c "SELECT 1" > /dev/null 2>&1; then
    check_result 0 "Database connection successful"
else
    check_result 1 "Cannot connect to database"
fi

# 4. Check pgvector extension
echo ""
echo "4. Checking pgvector extension..."
if docker-compose exec -T db psql -U postgres -d ai_agency -c "\dx" | grep -q vector; then
    check_result 0 "pgvector extension installed"
else
    check_result 1 "pgvector extension not found"
fi

# 5. Check tables exist
echo ""
echo "5. Checking database tables..."
TABLES=$(docker-compose exec -T db psql -U postgres -d ai_agency -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name IN ('tenants', 'runs', 'document_chunks')" 2>/dev/null | tr -d ' ')
if [ "$TABLES" = "3" ]; then
    check_result 0 "All 3 tables exist (tenants, runs, document_chunks)"
else
    check_result 1 "Expected 3 tables, found $TABLES"
    echo -e "${YELLOW}  Run migrations: DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency' alembic upgrade head${NC}"
fi

# 6. Check vector column
echo ""
echo "6. Checking vector column in document_chunks..."
if docker-compose exec -T db psql -U postgres -d ai_agency -c "\d document_chunks" | grep -q "vector(1536)"; then
    check_result 0 "Vector column exists with dimension 1536"
else
    check_result 1 "Vector column not found or wrong dimension"
fi

# 7. Check virtual environment
echo ""
echo "7. Checking virtual environment..."
if [ -d "venv" ]; then
    check_result 0 "Virtual environment exists"

    if [ -f "venv/bin/activate" ]; then
        check_result 0 "Virtual environment is valid"
    else
        check_result 1 "Virtual environment is broken"
    fi
else
    check_result 1 "Virtual environment not found"
    echo -e "${YELLOW}  Run: python3 -m venv venv && source venv/bin/activate && pip install -e .${NC}"
fi

# 8. Check Python imports
echo ""
echo "8. Checking Python imports..."
if [ -f "venv/bin/python" ]; then
    if venv/bin/python -c "from app.core import AIAgencyError, retry, BaseTool" 2>/dev/null; then
        check_result 0 "Core module imports work"
    else
        check_result 1 "Core module imports failed"
    fi
else
    echo -e "${YELLOW}⚠ Skipping import test (venv not found)${NC}"
fi

# 9. Check Alembic current revision
echo ""
echo "9. Checking Alembic migration status..."
if [ -f "venv/bin/alembic" ]; then
    CURRENT=$(DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency" venv/bin/alembic current 2>/dev/null | grep "001" || echo "")
    if [ -n "$CURRENT" ]; then
        check_result 0 "Alembic is at revision 001 (head)"
    else
        check_result 1 "Alembic migration not at head"
    fi
else
    echo -e "${YELLOW}⚠ Skipping Alembic check (not installed in venv)${NC}"
fi

# 10. Check documentation files
echo ""
echo "10. Checking documentation..."
DOCS=("docs/ARCHITECTURE.md" "docs/CODING_STANDARDS.md" "docs/DEPLOYMENT.md" "docs/CODE_REVIEW_CHECKLIST.md" "docs/WAVE1_REVIEW.md")
DOC_COUNT=0
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        DOC_COUNT=$((DOC_COUNT + 1))
    fi
done
if [ $DOC_COUNT -eq 5 ]; then
    check_result 0 "All 5 documentation files exist"
else
    check_result 1 "Expected 5 docs, found $DOC_COUNT"
fi

# Summary
echo ""
echo "===================================="
echo "Validation Summary"
echo "===================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Wave 1 infrastructure is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Fix CI (see docs/PRE_DEPLOYMENT_CHECKLIST.md)"
    echo "  2. Prepare credentials for Wave 2"
    echo "  3. Launch Wave 2 agents"
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS issue(s)${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
    exit 1
fi
