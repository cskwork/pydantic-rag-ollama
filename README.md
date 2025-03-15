# Pydantic AI RAG Example

A Retrieval-Augmented Generation (RAG) example using Pydantic AI, Ollama, and PostgreSQL with pgvector for semantic search.

## Overview

This project demonstrates how to build a RAG system that:
- Uses Ollama for local LLM inference and embeddings
- Stores documentation in PostgreSQL with vector search capabilities
- Handles question answering with contextual retrieval
- Utilizes Pydantic AI for agent-based operations

## Requirements

- Python 3.10+
- Docker Desktop for Windows
- PostgreSQL with pgvector extension
- Ollama with `nemotron-mini:latest` and `nomic-embed-text` models

## Setup

1. Install dependencies:
   ```bash
   pip install asyncpg httpx logfire ollama pydantic pydantic-ai
   ```

2. Setup PostgreSQL with pgvector using Docker (Windows):
   ```bash
   # Create a Docker network
   docker network create rag-network

   # Run PostgreSQL with pgvector
   docker run --name postgres-pgvector --network rag-network -e POSTGRES_PASSWORD=admin -e POSTGRES_USER=postgres -p 5432:5432 -d pgvector/pgvector:pg16
   
   # Create the database
   docker exec postgres-pgvector psql -U postgres -c "CREATE DATABASE pydantic_ai_rag;"
   
   # Verify pgvector extension is available
   docker exec postgres-pgvector psql -U postgres -d pydantic_ai_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. Setup Ollama using Docker (Windows):
   ```bash
   # Run Ollama
   docker run --name ollama -p 11434:11434 -v ollama:/root/.ollama -d ollama/ollama
   
   # Pull required models
   docker exec ollama ollama pull nemotron-mini:latest
   docker exec ollama ollama pull nomic-embed-text
   ```

4. Alternative: Manual Setup:
   - Ensure PostgreSQL is running (default: localhost:5432)
   - Install the `pgvector` extension manually
   - Create a database named `pydantic_ai_rag`
   - Download and run Ollama models locally

## Usage

### Build the search database:
```bash
python main-run.py build
```

### Run a search query:
```bash
python main-run.py search "How do I configure logfire to work with FastAPI?"
```

## Key Components

- `Agent` class from Pydantic AI for orchestrating the RAG workflow
- `retrieve` tool for semantic search using vector embeddings
- PostgreSQL with pgvector for efficient similarity search
- Ollama for local LLM inference and embedding generation
- Docker containers for easy setup and isolation

## Architecture

1. Question is passed to the agent
2. Agent generates embeddings for the query using Ollama
3. Vector search finds relevant documentation sections
4. LLM generates an answer based on retrieved context

## Configuration

Edit database connection parameters in the `database_connect` function to match your Docker setup:
```python
host = 'localhost'  # Use 'postgres-pgvector' if running in Docker network
port = '5432'
user = 'postgres'
password = 'admin'
```

### Docker Environment Variables

If you need to modify your Docker container settings:

```bash
# PostgreSQL environment variables
POSTGRES_USER=postgres
POSTGRES_PASSWORD=admin
POSTGRES_DB=pydantic_ai_rag

# Ollama environment variables
OLLAMA_HOST=0.0.0.0
OLLAMA_MODELS=/root/.ollama/models
```

## Logging

The application uses Logfire for structured logging. Configure with your own token:
```python
logfire.configure(send_to_logfire='if-token-present', token='your_token_here')
```

## Troubleshooting Docker Setup

### PostgreSQL Container Issues
- **Connection refused**: Ensure ports are correctly mapped with `-p 5432:5432`
- **pgvector extension not found**: Verify you're using `pgvector/pgvector:pg16` image
- **Database not created**: Run `docker exec postgres-pgvector psql -U postgres -c "CREATE DATABASE pydantic_ai_rag;"`

### Ollama Container Issues
- **Model download failures**: Check Docker volume permissions and network connection
- **API connection errors**: Verify the API is accessible at `http://localhost:11434/v1`
- **Memory issues**: Increase Docker Desktop resource allocation in settings

### Script Configuration
- When using Docker, update the database connection in script to match container names:
  ```python
  # If running the Python script outside Docker but connecting to Docker PostgreSQL
  host = 'localhost'
  port = '5432'
  
  # If running both in Docker with network
  host = 'postgres-pgvector'
  port = '5432'
  ```