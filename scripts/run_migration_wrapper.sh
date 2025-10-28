#!/bin/bash
set -e
set -x

echo "=== Starting migration wrapper script ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "DATABASE_URL set: ${DATABASE_URL:+YES}"

# Run the migration script
python /app/scripts/test_db_connection.py 2>&1

echo "=== Migration wrapper completed ==="
