"""RAG retrieval functions."""
import logging
from typing import List, Dict, Any, Optional

import asyncpg
import logfire

from embedding.service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


async def retrieve_documents(
    pool: asyncpg.Pool,
    query: str,
    limit: int = 8,
    embedding_service: Optional[EmbeddingService] = None,
) -> List[Dict[str, Any]]:
    """Retrieve documents based on semantic similarity.
    
    Args:
        pool (asyncpg.Pool): Database connection pool.
        query (str): The search query.
        limit (int, optional): Maximum number of results. Defaults to 8.
        embedding_service (Optional[EmbeddingService], optional): Embedding service.
            Defaults to None.
            
    Returns:
        List[Dict[str, Any]]: List of retrieved documents.
    """
    if embedding_service is None:
        embedding_service = get_embedding_service()
    
    with logfire.span('create embedding for {search_query=}', search_query=query):
        embedding = await embedding_service.get_embedding(query)
        embedding_json = embedding_service.embedding_to_json(embedding)
    
    rows = await pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT $2',
        embedding_json,
        limit,
    )
    
    # Convert to list of dictionaries
    return [dict(row) for row in rows]


def format_document_results(documents: List[Dict[str, Any]]) -> str:
    """Format retrieved documents as text.
    
    Args:
        documents (List[Dict[str, Any]]): Retrieved documents.
        
    Returns:
        str: Formatted text.
    """
    return '\n\n'.join(
        f'# {doc["title"]}\nDocumentation URL: {doc["url"]}\n\n{doc["content"]}\n'
        for doc in documents
    )
