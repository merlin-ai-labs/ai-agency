"""Seed database with initial data.

Creates tables and optionally seeds test data for development.

TODO:
- Implement table creation using SQLModel.metadata.create_all()
- Add sample tenant data
- Add sample run data
- Add command-line flags (--drop, --test-data, etc.)
"""

import asyncio
from sqlmodel import create_engine, SQLModel
from app.config import settings
from app.db.models import Run, Tenant, DocumentChunk
import structlog

logger = structlog.get_logger()


def create_tables():
    """
    Create all database tables.

    TODO:
    - Create engine from settings.database_url
    - Call SQLModel.metadata.create_all(engine)
    - Handle errors (table already exists, etc.)
    """
    logger.info("seed.create_tables", db_url=settings.database_url)

    # Stub implementation
    # engine = create_engine(settings.database_url)
    # SQLModel.metadata.create_all(engine)

    print("✓ Tables created (stub)")


def seed_test_data():
    """
    Seed test data for development.

    TODO:
    - Create sample tenant
    - Create sample runs (queued, completed)
    - Create sample document chunks
    """
    logger.info("seed.seed_test_data")

    # Stub implementation
    print("✓ Test data seeded (stub)")


def main():
    """
    Main entry point.

    TODO:
    - Parse command-line arguments
    - Create tables
    - Optionally seed test data
    """
    print("🌱 Seeding database...")

    create_tables()
    seed_test_data()

    print("✅ Database seeded successfully")


if __name__ == "__main__":
    main()
