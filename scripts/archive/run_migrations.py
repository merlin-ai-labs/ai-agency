#!/usr/bin/env python3
"""Run Alembic migrations with detailed logging."""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting migration script...")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")
logger.info(f"DATABASE_URL present: {'DATABASE_URL' in os.environ}")

# Check if alembic.ini exists
alembic_ini_path = os.path.join(os.getcwd(), 'alembic.ini')
logger.info(f"Looking for alembic.ini at: {alembic_ini_path}")
logger.info(f"alembic.ini exists: {os.path.exists(alembic_ini_path)}")

# List files in current directory
logger.info("Files in current directory:")
for item in os.listdir('.'):
    logger.info(f"  - {item}")

try:
    # Import and run alembic
    from alembic.config import Config
    from alembic import command

    logger.info("Alembic imports successful")

    # Create Alembic configuration
    alembic_cfg = Config("alembic.ini")
    logger.info("Alembic config created")

    # Run the upgrade
    logger.info("Running alembic upgrade head...")
    command.upgrade(alembic_cfg, "head")
    logger.info("Migration completed successfully!")

except Exception as e:
    logger.error(f"Migration failed with error: {e}", exc_info=True)
    sys.exit(1)
