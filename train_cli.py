#!/usr/bin/env python3
"""
CLI Training Interface for Fine-tuning Language Models
Provides an easy-to-use command line interface with presets and interactive configuration
"""

import argparse
import json
import os
import sys
from typing import Dict, Any
import subprocess

# Preset configurations for common use cases
PRESETS = {
    "qwen-python": {
        "name": "Qwen Python Coder",
        "description": "Fine-tune Qwen2.5-8B on Python code instructions",
        "config": {
            "model_name": "Qwen/Qwen2.5-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./qwen-python-coder",
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "quantization": "4bit"
        }
    },
    "qwen-general": {
        "name": "Qwen General Purpose",
        "description": "Fine-tune Qwen2.5-8B for general instruction following",
        "config": {
            "model_name": "Qwen/Qwen2.5-8B",
            "dataset_name": "tatsu-lab/alpaca",
            "output_dir": "./qwen-general",
            "max_seq_length": 2048,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-4,
            "lora_r": 64,
            "lora_alpha": 128,
            "quantization": "4bit"
        }
    },
    "llama-python": {
        "name": "Llama Python Coder",
        "description": "Fine-tune Llama 3.1 8B on Python code instructions",
        "config": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./llama-python-coder",
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "quantization": "4bit"
        }
    },
    "mistral-code": {
        "name": "Mistral Code Assistant",
        "description": "Fine-tune Mistral 7B for code generation",
        "config": {
            "model_name": "mistralai/Mistral-7B-v0.1",
            "dataset_name": "codeparrot/github-code-clean",
            "output_dir": "./mistral-code",
            "max_seq_length": 2048,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-4,
            "lora_r": 64,
            "lora_alpha": 128,
            "quantization": "4bit"
        }
    },
    "qwen-python-multi": {
        "name": "Qwen Python Coder (Multi-GPU)",
        "description": "Fine-tune Qwen2.5-8B on Python code with 2+ GPUs",
        "config": {
            "model_name": "Qwen/Qwen2.5-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./qwen-python-multi",
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,  # Higher batch size per GPU
            "gradient_accumulation_steps": 4,  # Reduced since we have more GPUs
            "learning_rate": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "quantization": "4bit",
            "use_multi_gpu": True,
            "gpu_ids": "0,1",  # Default to 2 GPUs
            "ddp_backend": "nccl"
        }
    },
    "qwen-python-4gpu": {
        "name": "Qwen Python Coder (4 GPUs)",
        "description": "Fine-tune Qwen2.5-8B on Python code with 4 GPUs",
        "config": {
            "model_name": "Qwen/Qwen2.5-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./qwen-python-4gpu",
            "max_seq_length": 2048,  # Longer sequences with more memory
            "num_train_epochs": 3,
            "per_device_train_batch_size": 6,  # Higher batch size per GPU
            "gradient_accumulation_steps": 2,  # Reduced since we have 4 GPUs
            "learning_rate": 3e-4,  # Slightly higher LR for larger batch
            "lora_r": 64,  # Higher rank for better quality
            "lora_alpha": 128,
            "quantization": "4bit",
            "use_multi_gpu": True,
            "gpu_ids": "0,1,2,3",  # 4 GPUs
            "ddp_backend": "nccl"
        }
    },
    "llama-python-multi": {
        "name": "Llama Python Coder (Multi-GPU)",
        "description": "Fine-tune Llama 3.1 8B on Python code with 2+ GPUs",
        "config": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./llama-python-multi",
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "quantization": "4bit",
            "use_multi_gpu": True,
            "gpu_ids": "0,1",
            "ddp_backend": "nccl"
        }
    }
}

# Popular datasets for different tasks
POPULAR_DATASETS = {
    "python_code": [
        "iamtarun/python_code_instructions_18k_alpaca",
        "codeparrot/github-code-clean",
        "bigcode/the-stack-python",
        "sahil2801/CodeAlpaca-20k"
    ],
    "general_instruction": [
        "tatsu-lab/alpaca",
        "yahma/alpaca-cleaned",
        "WizardLM/WizardLM_evol_instruct_70k",
        "Open-Orca/OpenOrca"
    ],
    "math": [
        "microsoft/orca-math-word-problems-200k",
        "meta-math/MetaMathQA",
        "TIGER-Lab/MathInstruct"
    ],
    "conversation": [
        "lmsys/chatbot_arena_conversations",
        "OpenAssistant/oasst1",
        "anthropic/hh-rlhf"
    ]
}

def list_presets():
    """List available presets"""
    print("🎯 Available Training Presets:")
    print("=" * 50)
    for key, preset in PRESETS.items():
        print(f"📦 {key}")
        print(f"   Name: {preset['name']}")
        print(f"   Description: {preset['description']}")
        print(f"   Model: {preset['config']['model_name']}")
        print(f"   Dataset: {preset['config']['dataset_name']}")
        print()

def list_datasets():
    """List popular datasets by category"""
    print("📊 Popular Datasets by Category:")
    print("=" * 50)
    for category, datasets in POPULAR_DATASETS.items():
        print(f"🏷️  {category.replace('_', ' ').title()}:")
        for dataset in datasets:
            print(f"   • {dataset}")
        print()

def interactive_config():
    """Interactive configuration builder"""
    print("🔧 Interactive Configuration Builder")
    print("=" * 50)
    
    config = {}
    
    # Model selection
    print("\n1. Model Selection:")
    print("   Popular models:")
    print("   • Qwen/Qwen2.5-8B (recommended for code)")
    print("   • meta-llama/Meta-Llama-3.1-8B")
    print("   • mistralai/Mistral-7B-v0.1")
    print("   • microsoft/DialoGPT-medium")
    
    model = input("\nEnter model name [Qwen/Qwen2.5-8B]: ").strip()
    config["model_name"] = model if model else "Qwen/Qwen2.5-8B"
    
    # Dataset selection
    print("\n2. Dataset Selection:")
    print("   For Python code: iamtarun/python_code_instructions_18k_alpaca")
    print("   For general chat: tatsu-lab/alpaca")
    print("   For math: microsoft/orca-math-word-problems-200k")
    
    dataset = input("\nEnter dataset name [iamtarun/python_code_instructions_18k_alpaca]: ").strip()
    config["dataset_name"] = dataset if dataset else "iamtarun/python_code_instructions_18k_alpaca"
    
    # Output directory
    output_dir = input("\nOutput directory [./trained-model]: ").strip()
    config["output_dir"] = output_dir if output_dir else "./trained-model"
    
    # Training parameters
    print("\n3. Training Parameters:")
    epochs = input("Number of epochs [3]: ").strip()
    config["num_train_epochs"] = int(epochs) if epochs else 3
    
    batch_size = input("Batch size per device [2]: ").strip()
    config["per_device_train_batch_size"] = int(batch_size) if batch_size else 2
    
    learning_rate = input("Learning rate [2e-4]: ").strip()
    config["learning_rate"] = float(learning_rate) if learning_rate else 2e-4
    
    # LoRA parameters
    print("\n4. LoRA Parameters:")
    lora_r = input("LoRA rank [32]: ").strip()
    config["lora_r"] = int(lora_r) if lora_r else 32
    
    lora_alpha = input("LoRA alpha [64]: ").strip()
    config["lora_alpha"] = int(lora_alpha) if lora_alpha else 64
    
    # Quantization
    print("\n5. Quantization:")
    print("   • 4bit (recommended for most GPUs)")
    print("   • 8bit (for larger GPUs)")
    print("   • none (for very large GPUs)")

    quant = input("Quantization type [4bit]: ").strip()
    config["quantization"] = quant if quant in ["4bit", "8bit", "none"] else "4bit"

    # Multi-GPU settings
    print("\n6. Multi-GPU Settings:")
    print("   Do you want to use multiple GPUs for training?")
    print("   This can significantly speed up training if you have multiple GPUs.")

    use_multi = input("Use multi-GPU training? [y/N]: ").strip().lower()
    if use_multi in ['y', 'yes']:
        config["use_multi_gpu"] = True

        print("\n   Available GPU configurations:")
        print("   • 2 GPUs: 0,1")
        print("   • 4 GPUs: 0,1,2,3")
        print("   • 8 GPUs: 0,1,2,3,4,5,6,7")
        print("   • Custom: specify your own")

        gpu_ids = input("GPU IDs [0,1]: ").strip()
        config["gpu_ids"] = gpu_ids if gpu_ids else "0,1"

        # Adjust batch size for multi-GPU
        gpu_count = len(config["gpu_ids"].split(','))
        recommended_batch = max(1, config["per_device_train_batch_size"] * 2)  # Increase batch size
        recommended_grad_acc = max(1, config["gradient_accumulation_steps"] // gpu_count)  # Reduce grad acc

        print(f"\n   With {gpu_count} GPUs, consider adjusting:")
        print(f"   • Batch size per device: {recommended_batch} (current: {config['per_device_train_batch_size']})")
        print(f"   • Gradient accumulation: {recommended_grad_acc} (current: {config['gradient_accumulation_steps']})")

        adjust = input("Apply recommended multi-GPU settings? [Y/n]: ").strip().lower()
        if adjust not in ['n', 'no']:
            config["per_device_train_batch_size"] = recommended_batch
            config["gradient_accumulation_steps"] = recommended_grad_acc
            print("   ✅ Applied multi-GPU optimizations")

        config["ddp_backend"] = "nccl"  # Default for GPU
    else:
        config["use_multi_gpu"] = False
        config["gpu_ids"] = "0"

    return config

def save_config(config: Dict[str, Any], filename: str):
    """Save configuration to file"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Configuration saved to {filename}")

def run_training(config: Dict[str, Any]):
    """Run training with the given configuration"""
    # Build command line arguments
    cmd = ["python", "qwen3_training.py"]
    
    for key, value in config.items():
        if key == "quantization":
            cmd.extend([f"--{key}", str(value)])
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print("🚀 Starting training with command:")
    print(" ".join(cmd))
    print()
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("🎉 Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="CLI Training Interface for Fine-tuning Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-presets                    # List available presets
  %(prog)s --preset qwen-python              # Use Qwen Python preset
  %(prog)s --interactive                     # Interactive configuration
  %(prog)s --list-datasets                   # List popular datasets
  %(prog)s --config my_config.json           # Use saved configuration
        """
    )
    
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--list-datasets", action="store_true", help="List popular datasets")
    parser.add_argument("--preset", type=str, choices=PRESETS.keys(), help="Use a preset configuration")
    parser.add_argument("--interactive", action="store_true", help="Interactive configuration builder")
    parser.add_argument("--config", type=str, help="Load configuration from JSON file")
    parser.add_argument("--save-config", type=str, help="Save configuration to file (don't run training)")
    parser.add_argument("--dry-run", action="store_true", help="Show command that would be run without executing")
    
    args = parser.parse_args()
    
    if args.list_presets:
        list_presets()
        return
    
    if args.list_datasets:
        list_datasets()
        return
    
    config = None
    
    if args.preset:
        if args.preset not in PRESETS:
            print(f"❌ Unknown preset: {args.preset}")
            print("Available presets:", list(PRESETS.keys()))
            sys.exit(1)
        config = PRESETS[args.preset]["config"].copy()
        print(f"📦 Using preset: {PRESETS[args.preset]['name']}")
        print(f"📝 {PRESETS[args.preset]['description']}")
    
    elif args.config:
        if not os.path.exists(args.config):
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"📁 Loaded configuration from {args.config}")
    
    elif args.interactive:
        config = interactive_config()
    
    else:
        print("❌ Please specify --preset, --config, --interactive, or use --help for options")
        sys.exit(1)
    
    if args.save_config:
        save_config(config, args.save_config)
        return
    
    if args.dry_run:
        print("🔍 Dry run - would execute:")
        cmd = ["python", "qwen3_training.py"]
        for key, value in config.items():
            if key == "quantization":
                cmd.extend([f"--{key}", str(value)])
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        print(" ".join(cmd))
        return
    
    # Show configuration summary
    print("\n📋 Training Configuration:")
    print("=" * 30)
    for key, value in config.items():
        print(f"{key:25}: {value}")
    print()
    
    confirm = input("🤔 Proceed with training? [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Training cancelled")
        return
    
    run_training(config)

if __name__ == "__main__":
    main()
