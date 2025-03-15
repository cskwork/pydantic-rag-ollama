import asyncio
import asyncpg
import httpx


async def test_postgres_connection():
    """Test connection to PostgreSQL server."""
    # Customize these values to match your PostgreSQL installation
    host = 'localhost'
    port = '5432'  # Standard PostgreSQL port
    user = 'postgres'  # Default PostgreSQL username
    password = 'admin'  # Replace with your actual password

    server_dsn = f'postgresql://{user}:{password}@{host}:{port}'

    print(f"Testing PostgreSQL connection to {host}:{port}...")
    try:
        conn = await asyncpg.connect(server_dsn)
        version = await conn.fetchval('SELECT version()')
        print(f"✅ Successfully connected to PostgreSQL")
        print(f"PostgreSQL version: {version}")

        # Check if pgvector extension is available
        try:
            # First create the extension if it doesn't exist
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')

            ext_exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            if ext_exists:
                print("✅ pgvector extension is installed")
            else:
                print("❌ pgvector extension is NOT installed")
                print("   You need to run: CREATE EXTENSION vector;")
        except Exception as e:
            print(f"❌ Error checking pgvector extension: {e}")

        await conn.close()
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("   Please check your connection parameters and that PostgreSQL is running")


async def test_ollama_connection():
    """Test connection to Ollama API."""
    ollama_url = "http://localhost:11434"
    ollama_v1_url = "http://localhost:11434/v1"

    print(f"\nTesting Ollama connection to {ollama_url}...")
    async with httpx.AsyncClient() as client:
        try:
            # Try the base API first
            response = await client.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"✅ Successfully connected to Ollama")
                print(f"Available models: {', '.join([m.get('name', '') for m in models])}")
            else:
                print(f"❌ Received status code {response.status_code} from Ollama")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama at {ollama_url}: {e}")

        # Also test the OpenAI compatible endpoint
        try:
            print(f"\nTesting OpenAI-compatible endpoint at {ollama_v1_url}...")
            response = await client.get(f"{ollama_v1_url}/models")
            if response.status_code == 200:
                print(f"✅ Successfully connected to OpenAI-compatible endpoint")
                print(f"Response: {response.text[:100]}...")
            else:
                print(f"❌ Received status code {response.status_code} from OpenAI-compatible endpoint")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"❌ Failed to connect to OpenAI-compatible endpoint at {ollama_v1_url}: {e}")
            print("   Note: Ollama may not have the OpenAI compatibility layer enabled")


async def main():
    await test_postgres_connection()
    await test_ollama_connection()


if __name__ == "__main__":
    asyncio.run(main())