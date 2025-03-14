from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import ollama
import pydantic_core
from pydantic import TypeAdapter
from typing_extensions import AsyncGenerator

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present', token='{replace-with-logfire-token}')
logfire.info('Hello, {place}!', place='World')
logfire.instrument_asyncpg()

@dataclass
class Deps:
    ollama_provider: OpenAIProvider
    pool: asyncpg.Pool


# Use the built-in OpenAIModel with Ollama provider
ollama_provider = OpenAIProvider(base_url='http://localhost:11434/v1')
ollama_model = OpenAIModel(model_name='llama3.2:1b', provider=ollama_provider)
agent = Agent(ollama_model, deps_type=Deps, instrument=True)

@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.
    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        # Using the OpenAI Provider but with Ollama base URL
        # embedding_response = await context.deps.ollama_provider.embeddings.create(
        #     input=search_query,
        #     model='nomic-embed-text',  # Choose an embedding model available in Ollama
        # )
        # embedding = embedding_response.data[0].embedding
        embedding_response = ollama.embed(
            model='nomic-embed-text',  # Specify the embedding model available in Ollama
            input=search_query
        )
        embedding = embedding_response['embeddings'][0]
        print(embedding)

    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    ollama_provider = OpenAIProvider(base_url='http://localhost:11434/v1')

    # Note: logfire instrumentation may need to be adjusted for Ollama

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(ollama_provider=ollama_provider, pool=pool)
        answer = await agent.run(question, deps=deps)
        print(answer.data)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################
# python -m main search "How do I configure logfire to work with FastAPI?"

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    ollama_provider = OpenAIProvider(base_url='http://localhost:11434/')

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        tasks = []
        for section in sections:
            tasks.append(
                asyncio.create_task(
                    insert_doc_section(sem, ollama_provider, pool, section)
                )
            )
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Add count verification here, after all tasks complete
        count = await pool.fetchval('SELECT COUNT(*) FROM doc_sections')
        logfire.info('Database build complete. Total records: {count}', count=count)
        print(f"Total records inserted: {count}")

        # Optionally, you can add a sample check to see some of the inserted records
        sample = await pool.fetch("SELECT url, title FROM doc_sections LIMIT 3")
        print("\nSample records:")
        for row in sample:
            print(f"URL: {row['url']}")
            print(f"Title: {row['title']}")


async def insert_doc_section(
    sem: asyncio.Semaphore,
    ollama_provider: OpenAIProvider,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            # embedding_response = await ollama_provider.embeddings.create(
            #     input=section.embedding_content(),
            #     model='nomic-embed-text',  # Choose an embedding model available in Ollama
            # )
            # embedding = embedding_response.data[0].embedding

            embedding_response = ollama.embed(
                model='nomic-embed-text',  # Specify the embedding model available in Ollama
                input=section.embedding_content()
            )
            # Debug the response structure
            print(f"Embedding response keys: {embedding_response}")

            # Try different potential keys based on Ollama's response structure
            embedding = embedding_response['embeddings'][0]


        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )


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


sessions_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
        create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    # Customize these values to match your PostgreSQL installation
    host = 'localhost'
    port = '5432'  # Default PostgreSQL port is 5432, you're using 54320
    user = 'postgres'
    password = 'admin'  # Replace with your actual password

    server_dsn = f'postgresql://{user}:{password}@{host}:{port}'
    database = 'pydantic_ai_rag'

    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
-- DROP TABLE doc_sections;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- Ollama embedding dimensions may vary based on the model
    -- Adjust vector dimension according to your model
    -- embedding vector(1536) NOT NULL
    -- Change from 1536 to 768 to match your model's output
    embedding vector(768) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)