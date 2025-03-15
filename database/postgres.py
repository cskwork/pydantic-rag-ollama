"""PostgreSQL database operations."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
import logfire

from config.settings import Settings, get_settings


@asynccontextmanager
async def database_connect(
    create_db: bool = False,
    settings: Optional[Settings] = None,
) -> AsyncGenerator[asyncpg.Pool, None]:
    """Connect to the database.
    
    Args:
        create_db (bool, optional): Whether to create the database if it doesn't exist.
            Defaults to False.
        settings (Optional[Settings], optional): Settings override. Defaults to None.
            
    Yields:
        AsyncGenerator[asyncpg.Pool, None]: Database connection pool.
    """
    if settings is None:
        settings = get_settings()
    
    db_settings = settings.database
    
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(db_settings.server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', db_settings.database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {db_settings.database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(db_settings.dsn)
    try:
        yield pool
    finally:
        await pool.close()


async def create_pool(settings: Optional[Settings] = None) -> asyncpg.Pool:
    """Create a database connection pool.
    
    Args:
        settings (Optional[Settings], optional): Settings override. Defaults to None.
            
    Returns:
        asyncpg.Pool: Database connection pool.
    """
    if settings is None:
        settings = get_settings()
    
    return await asyncpg.create_pool(settings.database.dsn)


async def create_schema(
    pool: asyncpg.Pool,
    vector_dimensions: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> None:
    """Create the database schema.
    
    Args:
        pool (asyncpg.Pool): Database connection pool.
        vector_dimensions (Optional[int], optional): Vector dimensions override.
            Defaults to None.
        settings (Optional[Settings], optional): Settings override. Defaults to None.
    """
    if settings is None:
        settings = get_settings()
    
    if vector_dimensions is None:
        vector_dimensions = settings.embedding_dimensions
    
    db_schema = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS doc_sections (
        id serial PRIMARY KEY,
        url text NOT NULL UNIQUE,
        title text NOT NULL,
        content text NOT NULL,
        embedding vector({vector_dimensions}) NOT NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding 
    ON doc_sections USING hnsw (embedding vector_l2_ops);
    """
    
    with logfire.span('create schema'):
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(db_schema)
