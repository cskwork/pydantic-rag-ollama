#!/usr/bin/env python
"""
Main application runner that uses the modularized components.
This file orchestrates the application flow without modifying existing files.
"""
from __future__ import annotations

import logging
# Standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Import configuration 
from config.settings import get_settings
# Import database components
from database.postgres import database_connect, create_schema
# Import embedding services
from embedding.service import get_embedding_service
# Import RAG components
from rag.retrieval import retrieve_documents, format_document_results
# Import provider components
from providers.base import create_provider
from providers.ollama import OllamaProvider
# Import utils
from utils import slugify
# Import necessary libraries
import logfire
import httpx
import pydantic_core
import asyncio
import sys
import ollama

# Import from pydantic_ai if available
try:
    from pydantic_ai import RunContext
    from pydantic_ai.agent import Agent
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    logger.warning("pydantic_ai not available, some features will be disabled")
    PYDANTIC_AI_AVAILABLE = False

# Borrowed from original code
from dataclasses import dataclass
from pydantic import TypeAdapter

# Configure logfire
def configure_logfire():
    """Configure logfire if token is available."""
    settings = get_settings()
    if settings.logfire_token:
        logfire.configure(
            send_to_logfire='if-token-present', 
            token=settings.logfire_token
        )
        logfire.info('Hello, {place}!', place='World')
        logfire.instrument_asyncpg()
    else:
        logger.info("Logfire token not found, skipping configuration")


@dataclass
class Deps:
    """Dependencies container for agent."""
    ollama_provider: OpenAIProvider
    pool: asyncpg.Pool


def initialize_agent():
    """Initialize the agent with the Ollama provider."""
    if not PYDANTIC_AI_AVAILABLE:
        logger.error("pydantic_ai is required for agent functionality")
        return None
    
    settings = get_settings()
    ollama_provider = OpenAIProvider(base_url=f'{settings.ollama.base_url}/v1')
    ollama_model = OpenAIModel(
        model_name=settings.ollama.completion_model, 
        provider=ollama_provider
    )
    return Agent(ollama_model, deps_type=Deps, instrument=True)


async def retrieve_tool(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.
    
    Args:
        context: The call context.
        search_query: The search query.
    """
    # Get the embedding service
    embedding_service = get_embedding_service()
    
    # Retrieve documents
    documents = await retrieve_documents(
        pool=context.deps.pool,
        query=search_query,
        embedding_service=embedding_service
    )
    
    # Format and return results
    return format_document_results(documents)


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    if not PYDANTIC_AI_AVAILABLE:
        logger.error("pydantic_ai is required for agent functionality")
        return
    
    settings = get_settings()
    ollama_provider = OpenAIProvider(base_url=f'{settings.ollama.base_url}/v1')
    agent = initialize_agent()
    
    if agent is None:
        logger.error("Failed to initialize agent")
        return
    
    # Register the retrieve tool
    if not hasattr(agent, 'retrieve'):
        agent.tool(retrieve_tool, name='retrieve')
    
    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(ollama_provider=ollama_provider, pool=pool)
        answer = await agent.run(question, deps=deps)
        print(answer.data)


# Data models for document sections
@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))


# Schema definition
DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    embedding vector(768) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


# URL for sample docs 
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)

async def insert_doc_section(
    sem: asyncio.Semaphore,
    provider,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    """Insert a document section into the database."""
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding_response = ollama.embed(
                model='nomic-embed-text',
                input=section.embedding_content()
            )
            embedding = embedding_response['embeddings'][0]

        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )


async def build_search_db():
    """Build the search database."""
    settings = get_settings()
    provider = create_provider(settings=settings)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    
    sessions_ta = TypeAdapter(list[DocsSection])
    sections = sessions_ta.validate_json(response.content)

    async with database_connect(True) as pool:
        await create_schema(pool)

        sem = asyncio.Semaphore(10)
        tasks = []
        for section in sections:
            tasks.append(
                asyncio.create_task(
                    insert_doc_section(sem, provider, pool, section)
                )
            )
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Add count verification
        count = await pool.fetchval('SELECT COUNT(*) FROM doc_sections')
        logfire.info('Database build complete. Total records: {count}', count=count)
        print(f"Total records inserted: {count}")

        # Sample check
        sample = await pool.fetch("SELECT url, title FROM doc_sections LIMIT 3")
        print("\nSample records:")
        for row in sample:
            print(f"URL: {row['url']}")
            print(f"Title: {row['title']}")


async def main():
    """Main entry point for the application."""
    configure_logfire()
    
    # Parse command line arguments
    action = sys.argv[1] if len(sys.argv) > 1 else None
    
    if action == 'build':
        await build_search_db()
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        await run_agent(q)
    else:
        print(
            'Usage: python main-run.py build|search [query]',
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
