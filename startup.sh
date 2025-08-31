#!/bin/bash
set -e

echo 'Waiting for Ollama to be ready...'
sleep 10

# Pull Ollama model
curl -X POST http://ollama:11434/api/pull -d '{"name":"llava:7b"}'

# Initialize models directory
echo 'Initializing models directory...'
python /app/init_models.py

# Start the application
echo 'Starting application...'
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload