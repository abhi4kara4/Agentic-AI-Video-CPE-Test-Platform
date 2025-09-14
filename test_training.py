#!/usr/bin/env python3
"""
Test script to verify PaddlePaddle training components work
"""

def test_paddle_training():
    """Test basic PaddlePaddle training setup"""
    try:
        print("Testing PaddlePaddle training components...")
        
        # Import components
        import paddle
        import paddle.nn as nn
        import paddle.optimizer as opt
        print(f"✅ PaddlePaddle {paddle.__version__} imported successfully")
        
        # Create simple model
        class TestModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2D(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2D((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 2)
                )
            
            def forward(self, x):
                return self.backbone(x)
        
        # Initialize model
        model = TestModel()
        print("✅ Test model created successfully")
        
        # Test optimizer
        optimizer = opt.Adam(learning_rate=0.001, parameters=model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        print("✅ Optimizer and loss function created")
        
        # Test forward pass
        data = paddle.rand([2, 3, 64, 64])  # Batch of 2 images
        labels = paddle.randint(0, 2, [2])  # Binary labels
        
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        print(f"✅ Forward pass successful - Loss: {loss.item():.4f}")
        
        # Test backward pass
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        print("✅ Backward pass and optimization successful")
        
        # Test model saving
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pdmodel', delete=False) as tmp:
            try:
                paddle.save(model.state_dict(), tmp.name)
                print(f"✅ Model saving successful - Size: {os.path.getsize(tmp.name)} bytes")
            finally:
                os.unlink(tmp.name)
        
        print("🎉 All PaddlePaddle training components work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ PaddlePaddle training test failed: {e}")
        return False

if __name__ == "__main__":
    test_paddle_training()