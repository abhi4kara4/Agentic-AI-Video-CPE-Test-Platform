#!/bin/bash
set -e

# Ensure Python can find our modules
export PYTHONPATH=/app:$PYTHONPATH

# Log the Python path for debugging
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Checking src/models directory:"
ls -la /app/src/models/ || echo "Models directory not found!"

# Execute the command passed to docker run
exec "$@"