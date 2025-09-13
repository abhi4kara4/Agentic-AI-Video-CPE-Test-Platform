# PaddleOCR Real Training Setup

This document outlines the changes made to enable real PaddleOCR training instead of simulation.

## Changes Made

### 1. Updated Dockerfile
- **Added PaddlePaddle dependencies**: System packages for OpenCV, image processing, and compilation tools
- **Environment variables**: Proper encoding and locale settings for PaddlePaddle
- **Staged installation**: Install dependencies in logical groups to avoid conflicts
- **Verification steps**: Post-installation checks to ensure PaddleOCR works correctly
- **Directory structure**: Created proper directories for models and manifests
- **Pre-trained models**: Download script to populate common models during build

### 2. Updated requirements.txt
Added PaddleOCR and required dependencies (compatible with Python 3.11):
```
paddlepaddle>=3.0.0
paddleocr>=2.7.0
paddlenlp>=2.6.0
shapely>=2.0.0
scikit-image>=0.21.0
lmdb>=1.4.0
tqdm>=4.66.0
attrdict>=2.0.1
openpyxl>=3.1.0
```

**Note**: Updated to PaddlePaddle 3.0+ which is compatible with Python 3.11

### 3. Updated PaddleOCR Trainer
- **Fixed import statement**: Proper PaddleOCR import
- **Better status reporting**: Clear indication when real training vs simulation
- **Proper detection**: Check for both PaddlePaddle and PaddleOCR availability

### 4. Created Pre-trained Model Downloader
- **Automatic model download**: Downloads English and Chinese models during build
- **Manifest creation**: Properly structures model registry
- **Error handling**: Graceful fallback if download fails

## How to Rebuild

### Option 1: Complete rebuild (recommended)
```bash
# Stop current container
docker-compose down

# Remove existing image
docker rmi agentic-ai-video-cpe-test-platform

# Rebuild with no cache to ensure fresh installation
docker-compose build --no-cache

# Start the container
docker-compose up -d
```

### Option 2: Force rebuild specific service
```bash
docker-compose build --no-cache ai-test-platform
docker-compose up -d ai-test-platform
```

## Verification

After rebuilding, check the logs during startup:
```bash
docker-compose logs ai-test-platform | grep -E "(PaddlePaddle|PaddleOCR|training)"
```

You should see:
- ✅ `PaddlePaddle version: 2.5.2`
- ✅ `PaddleOCR imported successfully`
- ✅ `PaddlePaddle and PaddleOCR are available. Real training enabled.`

Instead of:
- ❌ `Warning: PaddlePaddle not available. Training will be simulated.`

## Expected Build Time

The first build will take significantly longer (~10-20 minutes) due to:
- Installing system dependencies
- Downloading PaddlePaddle (~200MB)
- Installing PaddleOCR and dependencies (~500MB)
- Downloading pre-trained models (~100MB per language)

Subsequent builds will be faster due to Docker layer caching.

## Model Storage

Models will be stored in:
- **Pre-trained models**: `/app/Archive/paddleocr_models/`
- **Trained models**: `/app/volumes/trained_models/paddleocr/`
- **Model manifests**: JSON files tracking available models

## Testing Real Training

1. **Create a PaddleOCR dataset** with labeled text images
2. **Start training** - should see "Real training enabled" message
3. **Monitor progress** - actual PaddleOCR training metrics instead of simulation
4. **Check output** - real model files with proper weights instead of empty placeholders

The training will now use actual PaddleOCR algorithms and produce deployable models for production use.

## Troubleshooting

### Build Issues

**1. Python Version Compatibility**
- Issue: `ERROR: Could not find a version that satisfies the requirement paddlepaddle==X.X.X`
- Solution: The Dockerfile now uses `paddlepaddle>=3.0.0` which is compatible with Python 3.11

**2. ARM64 Architecture Issues**
- Issue: Some packages may not have ARM64 wheels
- Solution: The Dockerfile includes build tools to compile from source if needed

**3. Memory Issues During Build**
- Issue: Docker build runs out of memory
- Solution: Increase Docker memory limit or use staged installation approach

### Runtime Issues

**1. Import Errors**
```bash
# Check if PaddleOCR is properly installed
docker exec ai-test-platform python -c "import paddle; print(paddle.__version__)"
docker exec ai-test-platform python -c "from paddleocr import PaddleOCR; print('Success')"
```

**2. Model Download Failures**
- The system will gracefully fall back to simulation if pre-trained models can't be downloaded
- Training functionality will still work with user-provided data

**3. Performance Issues**
- First training run may be slower as PaddlePaddle optimizes for your hardware
- Subsequent runs will be faster due to compiled optimizations