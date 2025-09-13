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
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies first
RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 \
    pydantic-settings==2.1.0 python-multipart==0.0.6 opencv-python-headless==4.8.1.78 \
    ffmpeg-python==0.2.0 pillow==10.1.0 numpy==1.24.3 requests==2.31.0 aiohttp==3.9.1 \
    redis==5.0.1 python-dotenv==1.0.0 pyyaml==6.0.1 loguru==0.7.2 jinja2==3.1.2 tenacity==8.2.3

# Install AI/ML dependencies
RUN pip install --no-cache-dir ollama==0.1.7 ultralytics>=8.0.0 torch>=2.0.0 torchvision>=0.15.0

# Install PaddleOCR dependencies carefully
RUN pip install --no-cache-dir paddlepaddle>=3.0.0 && \
    pip install --no-cache-dir paddleocr>=2.7.0 && \
    pip install --no-cache-dir paddlex>=3.0.0 paddlenlp>=2.6.0 visualdl>=2.5.0 && \
    pip install --no-cache-dir shapely>=2.0.0 scikit-image>=0.21.0 lmdb>=1.4.0 \
    tqdm>=4.66.0 attrdict>=2.0.1 openpyxl>=3.1.0

# Install testing dependencies
RUN pip install --no-cache-dir pytest==7.4.3 pytest-bdd==6.1.1 pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 pytest-html==4.1.1 black==23.11.0 flake8==6.1.0 mypy==1.7.0 pre-commit==3.5.0

# Post-installation verification for PaddleOCR
RUN python -c "import paddle; print('PaddlePaddle version:', paddle.__version__)" && \
    python -c "from paddleocr import PaddleOCR; print('PaddleOCR imported successfully')" && \
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