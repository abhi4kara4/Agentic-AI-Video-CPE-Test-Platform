#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/app')

try:
    import paddle
    print('PaddlePaddle available:', paddle.__version__)
    PADDLE_AVAILABLE = True
except ImportError as e:
    print('PaddlePaddle import failed:', str(e))
    PADDLE_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    print('PaddleOCR available')
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print('PaddleOCR import failed:', str(e))
    PADDLEOCR_AVAILABLE = False

print(f'PADDLE_AVAILABLE: {PADDLE_AVAILABLE}')
print(f'PADDLEOCR_AVAILABLE: {PADDLEOCR_AVAILABLE}')

# Test the exact same logic as in paddleocr_trainer.py
print()
print('Testing trainer detection logic:')

try:
    import paddle
    print('Trainer - PaddlePaddle detected')
    trainer_paddle = True
except ImportError:
    print('Trainer - PaddlePaddle not detected')
    trainer_paddle = False

try:
    from paddleocr import PaddleOCR
    print('Trainer - PaddleOCR detected')
    trainer_paddleocr = True
except ImportError:
    print('Trainer - PaddleOCR not detected')
    trainer_paddleocr = False

print(f'Trainer logic - PADDLE_AVAILABLE: {trainer_paddle}')
print(f'Trainer logic - PADDLEOCR_AVAILABLE: {trainer_paddleocr}')

# Check what happens in trainer initialization
if not trainer_paddle:
    print('Trainer would show: Warning: PaddlePaddle not available. Training will be simulated.')
elif not trainer_paddleocr:
    print('Trainer would show: Warning: PaddleOCR not available. Training will be simulated.')
else:
    print('Trainer would show: PaddlePaddle and PaddleOCR are available. Real training enabled.')