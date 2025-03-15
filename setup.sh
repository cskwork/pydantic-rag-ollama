#!/bin/bash
# Setup script for pydantic-rag-ollama

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit the .env file with your configuration."
fi

echo "Setup complete! You can now run:"
echo "  python connection_test.py  # To test connections"
echo "  python main.py build       # To build the search database"
echo "  python main.py search      # To run a search query"
