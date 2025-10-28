#!/usr/bin/env python3
"""Test database connection and run migrations."""

import os
import sys
import logging

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=== Database Migration Test ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current directory: {os.getcwd()}")

        # Check DATABASE_URL
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Mask the password for logging
            masked_url = database_url
            if "@" in masked_url:
                parts = masked_url.split("@")
                if ":" in parts[0]:
                    user_pass = parts[0].split(":")
                    masked_url = f"{user_pass[0]}:****@{parts[1]}"
            logger.info(f"DATABASE_URL found: {masked_url}")
        else:
            logger.error("DATABASE_URL not set!")
            sys.exit(1)

        # Test psycopg connection
        logger.info("Testing psycopg connection...")
        import psycopg
        conn_params = database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                logger.info(f"Connected to PostgreSQL: {version[:50]}...")

        # Test pgvector
        logger.info("Testing pgvector extension...")
        with psycopg.connect(conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                result = cur.fetchone()
                if result:
                    logger.info(f"pgvector extension: {result[0]} version {result[1]}")
                else:
                    logger.warning("pgvector extension not found!")
                conn.commit()

        # Run Alembic migrations
        logger.info("Running Alembic migrations...")
        from alembic.config import Config
        from alembic import command

        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")

        logger.info("=== Migration completed successfully! ===")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
