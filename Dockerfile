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
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libx11-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for PaddlePaddle
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimized settings
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Post-installation verification for PaddleOCR
RUN python -c "import paddle; print('PaddlePaddle version:', paddle.__version__)" && \
    python -c "import paddleocr; print('PaddleOCR imported successfully')" && \
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Copy application code
COPY . .

# Download pre-trained PaddleOCR models
RUN python download_pretrained_models.py || echo "Pre-trained model download failed, continuing..."

# Create necessary directories
RUN mkdir -p logs screenshots reports training/models datasets testing \
    trained_models/paddleocr/det trained_models/paddleocr/rec trained_models/paddleocr/cls \
    /app/volumes/trained_models/paddleocr/det /app/volumes/trained_models/paddleocr/rec /app/volumes/trained_models/paddleocr/cls \
    Archive/paddleocr_models/det Archive/paddleocr_models/rec Archive/paddleocr_models/cls

# Create initial manifest files for PaddleOCR models
RUN echo '{"timestamp": 0, "models": {"det": {}, "rec": {}, "cls": {}}}' > Archive/paddleocr_models/manifest.json && \
    echo '{"timestamp": 0, "models": {"det": {}, "rec": {}, "cls": {}}}' > trained_models/paddleocr/manifest.json && \
    echo '{"timestamp": 0, "models": {"det": {}, "rec": {}, "cls": {}}}' > /app/volumes/trained_models/paddleocr/manifest.json

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