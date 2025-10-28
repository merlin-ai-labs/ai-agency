#!/bin/bash
# Validation script to check DevOps infrastructure setup

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "DevOps Infrastructure Validation"
echo "=========================================="
echo ""

ERRORS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
    else
        echo -e "${RED}✗${NC} $1/ - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to check executable
check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 (executable)"
    else
        echo -e "${YELLOW}⚠${NC} $1 (not executable)"
    fi
}

echo "Checking CI/CD files..."
check_dir ".github/workflows"
check_file ".github/workflows/ci.yml"
check_file ".github/workflows/deploy.yml"
echo ""

echo "Checking Database Migration files..."
check_file "alembic.ini"
check_dir "app/db/migrations"
check_file "app/db/migrations/env.py"
check_file "app/db/migrations/script.py.mako"
check_dir "app/db/migrations/versions"
check_file "app/db/migrations/versions/001_initial.py"
check_file "app/db/base.py"
echo ""

echo "Checking Docker files..."
check_file "docker-compose.yml"
check_file ".dockerignore"
check_file "Dockerfile"
echo ""

echo "Checking Development Tools..."
check_file ".pre-commit-config.yaml"
check_file "scripts/setup_dev.sh"
check_file "scripts/run_migrations.sh"
check_executable "scripts/setup_dev.sh"
check_executable "scripts/run_migrations.sh"
echo ""

echo "Checking Documentation..."
check_file "docs/DEPLOYMENT.md"
check_file "docs/DEVOPS_IMPLEMENTATION_SUMMARY.md"
echo ""

echo "Checking Environment Templates..."
check_file ".env.example"
check_file ".env.local"
check_file ".env.gcp"
check_file ".env.aws"
check_file ".env.azure"
echo ""

echo "Checking Configuration..."
check_file "pyproject.toml"
echo ""

# Check if required Python packages would be installed
echo "Checking Python dependencies in pyproject.toml..."
if grep -q "alembic" pyproject.toml; then
    echo -e "${GREEN}✓${NC} alembic dependency found"
else
    echo -e "${RED}✗${NC} alembic dependency missing"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "pre-commit" pyproject.toml; then
    echo -e "${GREEN}✓${NC} pre-commit dependency found"
else
    echo -e "${RED}✗${NC} pre-commit dependency missing"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "DevOps infrastructure is properly set up."
    echo ""
    echo "Next steps:"
    echo "  1. Run './scripts/setup_dev.sh' to set up development environment"
    echo "  2. Update .env with your API keys"
    echo "  3. Run 'docker-compose up' to start services"
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    echo "Please review the missing files above."
    exit 1
fi
echo "=========================================="
