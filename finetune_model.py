#!/usr/bin/env python3
"""
Fine-tune vision models for TV/STB screen recognition
Supports multiple training frameworks
"""
import os
import json
import yaml
from typing import Dict, Any

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import log

def generate_axolotl_config(model_name: str, dataset_path: str) -> Dict[str, Any]:
    """Generate Axolotl configuration for fine-tuning"""
    
    base_configs = {
        "llava:7b": {
            "base_model": "liuhaotian/llava-v1.6-mistral-7b",
            "model_type": "LlavaForCausalLM",
            "tokenizer_type": "LlamaTokenizer",
            "load_in_8bit": False,
            "load_in_4bit": True,  # Use 4-bit for efficiency
            "strict": False,
        },
        "llava-phi3": {
            "base_model": "microsoft/Phi-3-mini-4k-instruct",
            "model_type": "AutoModelForCausalLM", 
            "tokenizer_type": "AutoTokenizer",
            "load_in_4bit": True,
        },
        "moondream": {
            "base_model": "vikhyatk/moondream2",
            "model_type": "AutoModelForCausalLM",
            "tokenizer_type": "AutoTokenizer",
            "load_in_4bit": True,
        }
    }
    
    base = base_configs.get(model_name, base_configs["llava:7b"])
    
    config = {
        **base,
        
        # Dataset
        "datasets": [{
            "path": dataset_path,
            "type": "llava",
            "conversation": "llava_v1"
        }],
        
        # Training parameters optimized for TV/STB domain
        "num_epochs": 3,
        "micro_batch_size": 1,  # Small for CPU/limited GPU
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "eval_steps": 50,
        "save_steps": 200,
        "logging_steps": 10,
        
        # Optimization
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "weight_decay": 0.0,
        
        # LoRA configuration for efficient fine-tuning
        "adapter": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        
        # Output
        "output_dir": f"./models/{model_name}-tv-stb-ft",
        "gradient_checkpointing": True,
        "flash_attention": True,
        
        # Special tokens for TV domain
        "special_tokens": {
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        },
        
        # Wandb logging (optional)
        "wandb_project": "tv-stb-vision-finetune",
        "wandb_watch": "false",
        "wandb_run_id": f"{model_name}-ft-tv",
    }
    
    return config

def generate_training_script(model_name: str, config: Dict[str, Any]) -> str:
    """Generate training script"""
    
    script = f"""#!/bin/bash
# Fine-tuning script for {model_name} on TV/STB dataset

# Install requirements
pip install -U transformers accelerate peft bitsandbytes
pip install git+https://github.com/OpenAccess-AI-Collective/axolotl.git

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your GPU
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"

# Run training
accelerate launch -m axolotl.cli.train config.yaml

# Merge LoRA weights back to base model
python -m axolotl.cli.merge_lora config.yaml

# Quantize for deployment (optional)
python quantize_model.py \\
    --model_path ./models/{model_name}-tv-stb-ft \\
    --quantization q4_K_M \\
    --output ./models/{model_name}-tv-stb-q4

echo "Fine-tuning complete!"
echo "Model saved to: ./models/{model_name}-tv-stb-ft"
echo "Quantized model: ./models/{model_name}-tv-stb-q4"
"""
    
    return script

def generate_ollama_modelfile(model_name: str, model_path: str) -> str:
    """Generate Ollama Modelfile for deployment"""
    
    modelfile = f"""# Modelfile for TV/STB optimized {model_name}
FROM {model_path}

# Model parameters optimized for TV screen analysis
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 500

# System prompt for TV/STB domain
SYSTEM \"\"\"You are a specialized AI for analyzing TV and Set-Top Box screens. 
You can identify:
- Screen types (home, app rail, content, errors)
- Application names and states
- UI elements and navigation
- Video playback status
- Screen anomalies

Always respond with structured JSON when analyzing screens.\"\"\"

# Example template
TEMPLATE \"\"\"{{{{ if .System }}}}{{ .System }}{{{{ end }}}}
{{{{ if .Prompt }}}}Human: {{ .Prompt }}{{{{ end }}}}
Assistant: I'll analyze this TV screen for you.
{{{{ .Response }}}}\"\"\"
"""
    
    return modelfile

def create_validation_set(dataset_path: str, split_ratio: float = 0.8):
    """Split dataset into train/validation sets"""
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Shuffle and split
    import random
    random.shuffle(data)
    
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save splits
    base_path = os.path.dirname(dataset_path)
    
    with open(os.path.join(base_path, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(base_path, "val.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    log.info(f"✓ Created train set: {len(train_data)} examples")
    log.info(f"✓ Created validation set: {len(val_data)} examples")
    
    return len(train_data), len(val_data)

def estimate_training_requirements(model_name: str, dataset_size: int) -> Dict[str, Any]:
    """Estimate compute requirements for fine-tuning"""
    
    model_sizes = {
        "llava:7b": {"params": 7e9, "base_vram": 14, "train_vram": 24},
        "llava-phi3": {"params": 3.8e9, "base_vram": 8, "train_vram": 16},
        "moondream": {"params": 1.6e9, "base_vram": 4, "train_vram": 8},
    }
    
    model_info = model_sizes.get(model_name, model_sizes["llava:7b"])
    
    # Estimate with LoRA (reduces requirements by ~70%)
    lora_vram = model_info["train_vram"] * 0.3
    
    # Time estimates (rough)
    time_per_epoch = dataset_size * 0.5  # seconds per sample
    total_time = time_per_epoch * 3 / 3600  # 3 epochs, convert to hours
    
    return {
        "model_parameters": f"{model_info['params']/1e9:.1f}B",
        "vram_required": f"{lora_vram:.1f}GB (with LoRA)",
        "vram_full": f"{model_info['train_vram']}GB (full fine-tuning)",
        "estimated_time": f"{total_time:.1f} hours",
        "recommended_gpu": "RTX 3090 or better" if lora_vram > 12 else "RTX 3070 or better",
        "cpu_training": "Possible but 10-20x slower" if model_info['params'] < 4e9 else "Not recommended"
    }

def main():
    """Main fine-tuning setup"""
    log.info("🎯 TV/STB Vision Model Fine-Tuning Setup")
    log.info("="*60)
    
    # Check for dataset
    dataset_path = "training_data/dataset.json"
    if not os.path.exists(dataset_path):
        log.error("No dataset found! Run prepare_training_data.py first")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    log.info(f"✓ Found dataset with {len(dataset)} examples")
    
    # Select model
    print("\n🤖 Select model to fine-tune:")
    print("1. llava:7b - Best accuracy (needs good GPU)")
    print("2. llava-phi3 - Balanced (moderate GPU)")
    print("3. moondream - Fastest (CPU possible)")
    
    choice = input("\nYour choice: ").strip()
    model_map = {"1": "llava:7b", "2": "llava-phi3", "3": "moondream"}
    model_name = model_map.get(choice, "llava-phi3")
    
    # Create train/val split
    log.info("\n📊 Creating train/validation split...")
    train_size, val_size = create_validation_set(dataset_path)
    
    # Estimate requirements
    log.info("\n💻 Compute Requirements:")
    reqs = estimate_training_requirements(model_name, train_size)
    for key, value in reqs.items():
        log.info(f"  {key}: {value}")
    
    # Generate configurations
    log.info("\n📝 Generating training configurations...")
    
    # Axolotl config
    config = generate_axolotl_config(model_name, "training_data/train.json")
    config_path = "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"✓ Created Axolotl config: {config_path}")
    
    # Training script
    script = generate_training_script(model_name, config)
    script_path = "run_finetuning.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    log.info(f"✓ Created training script: {script_path}")
    
    # Ollama Modelfile
    modelfile = generate_ollama_modelfile(model_name, f"./models/{model_name}-tv-stb-ft")
    modelfile_path = "Modelfile.tv"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    log.info(f"✓ Created Ollama Modelfile: {modelfile_path}")
    
    # Instructions
    log.info("\n🚀 Next Steps:")
    log.info("1. Review and adjust training_config.yaml if needed")
    log.info("2. Run fine-tuning:")
    log.info(f"   ./run_finetuning.sh")
    log.info("3. After training, create Ollama model:")
    log.info(f"   ollama create tv-{model_name} -f Modelfile.tv")
    log.info("4. Test the fine-tuned model:")
    log.info(f"   ollama run tv-{model_name}")
    
    # Quick test option
    if input("\n🧪 Generate test script for fine-tuned model? (y/n): ").lower() == 'y':
        test_script = f"""#!/usr/bin/env python3
import ollama
import base64
from PIL import Image
import sys

# Load test image
image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
with open(image_path, 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Test with fine-tuned model
response = ollama.chat(
    model='tv-{model_name}',
    messages=[{{
        'role': 'user',
        'content': 'Analyze this TV screen',
        'images': [image_data]
    }}]
)

print(response['message']['content'])
"""
        
        with open("test_finetuned.py", 'w') as f:
            f.write(test_script)
        os.chmod("test_finetuned.py", 0o755)
        log.info("✓ Created test script: test_finetuned.py")

if __name__ == "__main__":
    main()