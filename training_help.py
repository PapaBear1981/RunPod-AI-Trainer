#!/usr/bin/env python3
"""
Training Help and Quick Start Guide
Comprehensive guide for using the CLI training system
"""

import os
import sys

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section"""
    print(f"\n📋 {title}")
    print("-" * 40)

def show_quick_start():
    """Show quick start guide"""
    print_header("QUICK START GUIDE")
    
    print("""
🚀 Get started with AI training in 3 easy steps:

1️⃣  List available presets:
    python train_cli.py --list-presets

2️⃣  Start training with a preset:
    python train_cli.py --preset qwen-python

3️⃣  Or use interactive mode:
    python train_cli.py --interactive

That's it! The system will handle all the complex configuration for you.
""")

def show_presets():
    """Show information about presets"""
    print_header("AVAILABLE PRESETS")
    
    print("""
📦 qwen-python
   • Best for: Python code generation and assistance
   • Model: Qwen2.5-8B (8 billion parameters)
   • Dataset: 18k Python code instructions
   • Memory: Optimized for 24GB+ VRAM
   • Training time: ~2-3 hours on A40

📦 qwen-general  
   • Best for: General conversation and instruction following
   • Model: Qwen2.5-8B
   • Dataset: Alpaca instruction dataset
   • Memory: Optimized for 24GB+ VRAM
   • Training time: ~3-4 hours on A40

📦 llama-python
   • Best for: Python coding with Llama architecture
   • Model: Llama 3.1 8B
   • Dataset: 18k Python code instructions
   • Memory: Optimized for 24GB+ VRAM
   • Training time: ~2-3 hours on A40

📦 mistral-code
   • Best for: Multi-language code generation
   • Model: Mistral 7B
   • Dataset: GitHub code repository
   • Memory: Optimized for 16GB+ VRAM
   • Training time: ~2-3 hours on A40
""")

def show_datasets():
    """Show popular datasets"""
    print_header("POPULAR DATASETS")
    
    print("""
🐍 Python Code Datasets:
   • iamtarun/python_code_instructions_18k_alpaca (recommended)
   • codeparrot/github-code-clean
   • bigcode/the-stack-python
   • sahil2801/CodeAlpaca-20k

💬 General Instruction Datasets:
   • tatsu-lab/alpaca (recommended)
   • yahma/alpaca-cleaned
   • WizardLM/WizardLM_evol_instruct_70k
   • Open-Orca/OpenOrca

🧮 Math Datasets:
   • microsoft/orca-math-word-problems-200k
   • meta-math/MetaMathQA
   • TIGER-Lab/MathInstruct

🗣️ Conversation Datasets:
   • lmsys/chatbot_arena_conversations
   • OpenAssistant/oasst1
   • anthropic/hh-rlhf
""")

def show_commands():
    """Show all available commands"""
    print_header("COMMAND REFERENCE")
    
    print_section("Training Commands")
    print("""
# List available presets
python train_cli.py --list-presets

# Use a preset configuration
python train_cli.py --preset qwen-python

# Interactive configuration builder
python train_cli.py --interactive

# Load configuration from file
python train_cli.py --config my_config.json

# Save configuration without training
python train_cli.py --preset qwen-python --save-config my_config.json

# Dry run (show command without executing)
python train_cli.py --preset qwen-python --dry-run
""")
    
    print_section("Configuration Management")
    print("""
# List saved configurations
python config_manager.py list

# Save current configuration
python config_manager.py save my_config config.json

# Load a saved configuration
python config_manager.py load my_config

# Compare two configurations
python config_manager.py compare config1 config2

# Copy a configuration
python config_manager.py copy source_config new_config

# Delete a configuration
python config_manager.py delete old_config

# Create template configurations
python config_manager.py templates
""")
    
    print_section("Direct Training (Advanced)")
    print("""
# Train with specific parameters
python qwen3_training.py \\
    --model_name "Qwen/Qwen2.5-8B" \\
    --dataset_name "iamtarun/python_code_instructions_18k_alpaca" \\
    --output_dir "./my-model" \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 2 \\
    --learning_rate 2e-4 \\
    --lora_r 32 \\
    --quantization 4bit

# Load configuration from file
python qwen3_training.py --config_file my_config.json

# Save current configuration
python qwen3_training.py --save_config my_config.json
""")

def show_examples():
    """Show practical examples"""
    print_header("PRACTICAL EXAMPLES")
    
    print_section("Example 1: Quick Python Coder")
    print("""
# Train a Python coding assistant (easiest way)
python train_cli.py --preset qwen-python

# This will:
# - Use Qwen2.5-8B model
# - Train on Python code instructions
# - Use 4-bit quantization
# - Save to ./qwen-python-coder/
# - Take about 2-3 hours on A40 GPU
""")
    
    print_section("Example 2: Custom Configuration")
    print("""
# Create custom configuration interactively
python train_cli.py --interactive

# Save the configuration for later
python train_cli.py --interactive --save-config my_custom.json

# Use the saved configuration
python train_cli.py --config my_custom.json
""")
    
    print_section("Example 3: Experiment with Different Models")
    print("""
# Try different models with same dataset
python train_cli.py --preset qwen-python --save-config qwen_base.json
python train_cli.py --preset llama-python --save-config llama_base.json

# Compare the configurations
python config_manager.py compare qwen_base llama_base

# Modify and test
python config_manager.py copy qwen_base qwen_experiment
# Edit qwen_experiment.json manually
python train_cli.py --config qwen_experiment.json
""")

def show_troubleshooting():
    """Show troubleshooting guide"""
    print_header("TROUBLESHOOTING")
    
    print_section("Common Issues")
    print("""
❌ Out of Memory Error:
   • Reduce batch size: --per_device_train_batch_size 1
   • Increase gradient accumulation: --gradient_accumulation_steps 16
   • Use 4-bit quantization: --quantization 4bit
   • Reduce sequence length: --max_seq_length 512

❌ Model Not Found:
   • Check model name on HuggingFace Hub
   • Ensure you have internet connection
   • Try: huggingface-cli login (if private model)

❌ Dataset Not Found:
   • Check dataset name on HuggingFace Hub
   • Ensure dataset has correct format
   • Try: python train_cli.py --list-datasets

❌ Training Too Slow:
   • Increase batch size if you have VRAM
   • Reduce max_seq_length
   • Use fewer epochs for testing
   • Enable gradient checkpointing: --gradient_checkpointing

❌ Poor Results:
   • Increase LoRA rank: --lora_r 64
   • Increase learning rate: --learning_rate 5e-4
   • Train for more epochs: --num_train_epochs 5
   • Use larger dataset or better quality data
""")
    
    print_section("Memory Guidelines")
    print("""
🖥️  GPU Memory Requirements:

16GB VRAM (RTX 4090):
   • Use 4-bit quantization
   • Batch size: 1-2
   • Sequence length: 512-1024
   • LoRA rank: 16-32

24GB VRAM (RTX 4090 Ti):
   • Use 4-bit quantization
   • Batch size: 2-4
   • Sequence length: 1024-2048
   • LoRA rank: 32-64

48GB VRAM (A40/A100):
   • Use 4-bit or no quantization
   • Batch size: 4-8
   • Sequence length: 2048-4096
   • LoRA rank: 64-128
""")

def show_menu():
    """Show interactive menu"""
    while True:
        print_header("AI TRAINING HELP SYSTEM")
        print("""
Choose a topic:

1️⃣  Quick Start Guide
2️⃣  Available Presets
3️⃣  Popular Datasets
4️⃣  Command Reference
5️⃣  Practical Examples
6️⃣  Troubleshooting
7️⃣  Exit

""")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            show_quick_start()
        elif choice == '2':
            show_presets()
        elif choice == '3':
            show_datasets()
        elif choice == '4':
            show_commands()
        elif choice == '5':
            show_examples()
        elif choice == '6':
            show_troubleshooting()
        elif choice == '7':
            print("\n👋 Happy training!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7.")
        
        input("\nPress Enter to continue...")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        topic = sys.argv[1].lower()
        if topic == 'quick':
            show_quick_start()
        elif topic == 'presets':
            show_presets()
        elif topic == 'datasets':
            show_datasets()
        elif topic == 'commands':
            show_commands()
        elif topic == 'examples':
            show_examples()
        elif topic == 'troubleshooting':
            show_troubleshooting()
        else:
            print(f"❌ Unknown topic: {topic}")
            print("Available topics: quick, presets, datasets, commands, examples, troubleshooting")
    else:
        show_menu()

if __name__ == "__main__":
    main()
