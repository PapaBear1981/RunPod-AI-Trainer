# 🎯 AI Training CLI System Overview

You now have a comprehensive CLI-based training system that makes it super easy to configure and run AI training on RunPod! Here's what I've built for you:

## 🚀 What's New

Instead of editing Python code every time you want to change models, datasets, or parameters, you now have:

1. **Easy-to-use presets** for common training scenarios
2. **Interactive configuration builder** that walks you through options
3. **Command-line interface** for all training parameters
4. **Configuration management** to save and reuse settings
5. **RunPod-optimized workflows** with monitoring and packaging

## 📦 Package Contents

I've created `qwen3_training_cli_package.tar.gz` which contains:

### Core Files
- `qwen3_training.py` - Enhanced with full CLI support
- `train_cli.py` - User-friendly training interface with presets
- `config_manager.py` - Save, load, and compare configurations
- `training_help.py` - Comprehensive help and troubleshooting
- `runpod_setup_script.sh` - Complete RunPod setup with CLI integration

### RunPod-Specific
- `train_runpod.sh` - RunPod-optimized CLI launcher (created during setup)
- `RUNPOD_README.md` - Complete documentation for RunPod
- `QUICK_START.sh` - One-click setup script

## 🎮 How to Use on RunPod

### 1. Deploy the Package
```bash
# Upload qwen3_training_cli_package.tar.gz to your RunPod instance
tar -xzf qwen3_training_cli_package.tar.gz
./QUICK_START.sh
```

### 2. Quick Training (Easiest Way)
```bash
# Check your GPU and get recommendations
./train_runpod.sh gpu-check

# Start training with a preset
./train_runpod.sh qwen-python    # Python code assistant
./train_runpod.sh qwen-general   # General chat assistant
./train_runpod.sh llama-python   # Llama-based Python coder
```

### 3. Interactive Configuration
```bash
./train_runpod.sh interactive
```
This walks you through:
- Model selection (Qwen, Llama, Mistral, etc.)
- Dataset selection (Python code, general chat, math, etc.)
- Training parameters (epochs, batch size, learning rate)
- LoRA settings (rank, alpha, dropout)
- Quantization options (4-bit, 8-bit, none)

### 4. Advanced Usage
```bash
# Use saved configuration
./train_runpod.sh config my_config.json

# Direct parameter specification
python3 train_cli.py \
    --model_name "Qwen/Qwen2.5-8B" \
    --dataset_name "iamtarun/python_code_instructions_18k_alpaca" \
    --num_train_epochs 5 \
    --learning_rate 3e-4 \
    --lora_r 64
```

## 🎯 Available Presets

### qwen-python (Recommended)
- **Model:** Qwen2.5-8B
- **Dataset:** Python code instructions (18k samples)
- **Best for:** Python coding assistance
- **Memory:** 24GB+ VRAM
- **Time:** ~2-3 hours on A40

### qwen-general
- **Model:** Qwen2.5-8B  
- **Dataset:** Alpaca instructions
- **Best for:** General conversation
- **Memory:** 24GB+ VRAM
- **Time:** ~3-4 hours on A40

### llama-python
- **Model:** Llama 3.1 8B
- **Dataset:** Python code instructions
- **Best for:** Llama-based Python coding
- **Memory:** 24GB+ VRAM
- **Time:** ~2-3 hours on A40

### mistral-code
- **Model:** Mistral 7B
- **Dataset:** GitHub code
- **Best for:** Multi-language coding
- **Memory:** 16GB+ VRAM
- **Time:** ~2-3 hours on A40

### Memory-Optimized Variants
- **qwen-small:** For 16GB VRAM (reduced batch size, shorter sequences)
- **qwen-large:** For 48GB+ VRAM (larger batches, longer sequences)

## 🔧 Configuration Management

### Save Configurations
```bash
# Save a preset as a custom config
./train_runpod.sh qwen-python --save-config my_python_config.json

# Save interactive configuration
./train_runpod.sh interactive --save-config my_custom.json
```

### Manage Saved Configs
```bash
python3 config_manager.py list                    # List all saved configs
python3 config_manager.py compare config1 config2 # Compare two configs
python3 config_manager.py copy source target      # Copy a config
python3 config_manager.py delete old_config       # Delete a config
```

## 📊 Monitoring & Management

### Real-time Monitoring
```bash
./train_runpod.sh monitor
```
Shows:
- GPU utilization and memory usage
- CPU and RAM usage  
- Temperature monitoring
- Training progress

### Log Management
```bash
./train_runpod.sh logs      # View recent training logs
./train_runpod.sh cleanup   # Clean up old files
```

### Model Packaging
```bash
./train_runpod.sh package   # Package trained model for download
```
Creates a compressed archive you can transfer with `runpodctl send`.

## 🎛️ All Available Parameters

The CLI system supports all these parameters:

### Model Settings
- `--model_name` - HuggingFace model (Qwen, Llama, Mistral, etc.)
- `--max_seq_length` - Maximum sequence length

### LoRA Settings  
- `--lora_r` - LoRA rank (16, 32, 64, 128)
- `--lora_alpha` - LoRA alpha (usually 2x rank)
- `--lora_dropout` - Dropout rate

### Quantization
- `--quantization` - none, 4bit, 8bit
- `--bnb_4bit_compute_dtype` - float16, bfloat16
- `--bnb_4bit_quant_type` - fp4, nf4

### Training Parameters
- `--num_train_epochs` - Number of epochs
- `--per_device_train_batch_size` - Batch size per GPU
- `--gradient_accumulation_steps` - Gradient accumulation
- `--learning_rate` - Learning rate
- `--weight_decay` - Weight decay
- `--warmup_ratio` - Warmup ratio

### Dataset Settings
- `--dataset_name` - HuggingFace dataset name
- `--dataset_split` - train, validation, test
- `--max_samples` - Limit number of samples

## 🛠️ Troubleshooting

### Memory Issues
```bash
# Check GPU memory
./train_runpod.sh gpu-check

# Use memory-optimized preset
./train_runpod.sh qwen-small

# Or adjust parameters manually
python3 train_cli.py --preset qwen-python \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512
```

### Performance Issues
```bash
# Use larger batches if you have memory
--per_device_train_batch_size 4

# Enable gradient checkpointing
--gradient_checkpointing

# Use 4-bit quantization
--quantization 4bit
```

## 🎯 Why This is Better

**Before:** You had to edit Python code every time you wanted to change:
- Model (Qwen vs Llama vs Mistral)
- Dataset (Python vs general vs math)
- Training parameters (epochs, batch size, learning rate)
- LoRA settings (rank, alpha)
- Quantization options

**Now:** You can:
- Use presets for common scenarios: `./train_runpod.sh qwen-python`
- Configure interactively: `./train_runpod.sh interactive`
- Save and reuse configurations
- Compare different setups
- Monitor training in real-time
- Package models for easy transfer

## 🚀 Next Steps

1. **Upload the package** to your RunPod instance
2. **Run the demo** locally: `python3 demo_cli.py`
3. **Try the quick start** on RunPod: `./QUICK_START.sh`
4. **Experiment with presets** to find what works best for your use case
5. **Save successful configurations** for future training runs

The system is designed to be portable - you can use the same configurations across different RunPod instances, and everything is optimized for the A40 GPUs you prefer.

Happy training! 🎉
