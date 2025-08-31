FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs screenshots reports training/models datasets testing

# Backup models directory (in case volume mount overwrites it)
RUN cp -r /app/src/models /app/models_backup || true

# Ensure Python can find our modules
ENV PYTHONPATH=/app:$PYTHONPATH

# Copy startup scripts
COPY startup.sh /app/startup.sh
COPY init_models.py /app/init_models.py
RUN chmod +x /app/startup.sh

# Expose API port
EXPOSE 8000

# Default command
CMD ["/app/startup.sh"]