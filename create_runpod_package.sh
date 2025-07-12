#!/bin/bash

# Create RunPod Training Package with CLI System
# This script packages everything needed for RunPod deployment

set -e

PACKAGE_NAME="qwen3_training_cli_package.tar.gz"
TEMP_DIR="qwen3_training_cli_temp"

echo "📦 Creating RunPod Training Package with CLI System..."

# Clean up any existing temp directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "📋 Copying core training files..."
# Core training files
cp qwen3_training.py "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"
cp test_setup.py "$TEMP_DIR/"
cp runpod_setup_script.sh "$TEMP_DIR/"

echo "🖥️ Copying CLI system files..."
# CLI system files
cp train_cli.py "$TEMP_DIR/"
cp config_manager.py "$TEMP_DIR/"
cp training_help.py "$TEMP_DIR/"
cp train.sh "$TEMP_DIR/"
cp train_multi_gpu.sh "$TEMP_DIR/"

echo "📚 Creating README for RunPod..."
# Create comprehensive README for RunPod
cat > "$TEMP_DIR/RUNPOD_README.md" << 'EOF'
# 🚀 AI Training CLI System for RunPod

This package provides a comprehensive CLI system for training language models on RunPod with easy-to-use presets and configuration management.

## 🎯 Quick Start

1. **Upload and extract this package to your RunPod instance**
2. **Run the setup script:**
   ```bash
   chmod +x runpod_setup_script.sh
   ./runpod_setup_script.sh
   ```
3. **Start training with a preset:**
   ```bash
   ./train_runpod.sh qwen-python
   ```

## 🎮 CLI Commands

### Quick Commands
```bash
./train_runpod.sh gpu-check      # Check GPU and get recommendations
./train_runpod.sh presets        # List all available presets
./train_runpod.sh interactive    # Interactive configuration builder
./train_runpod.sh help          # Full help system
```

### Single GPU Training Presets
```bash
./train_runpod.sh qwen-python    # Qwen Python coder (recommended)
./train_runpod.sh qwen-general   # Qwen general assistant
./train_runpod.sh llama-python   # Llama Python coder
./train_runpod.sh mistral-code   # Mistral code assistant
./train_runpod.sh qwen-small     # Memory-optimized (16GB)
./train_runpod.sh qwen-large     # High-memory version (48GB+)
```

### Multi-GPU Training Presets (2+ GPUs)
```bash
./train_multi_gpu.sh detect              # Detect available GPUs
./train_multi_gpu.sh recommend           # Get multi-GPU recommendations
./train_multi_gpu.sh qwen-python-multi   # 2 GPU Python coder
./train_multi_gpu.sh qwen-python-4gpu    # 4 GPU Python coder (high memory)
./train_multi_gpu.sh llama-python-multi  # 2 GPU Llama Python coder
./train_multi_gpu.sh custom 0,1,2,3      # Custom GPU configuration
```

### Monitoring & Management
```bash
./train_runpod.sh monitor        # Real-time system monitoring
./train_runpod.sh logs           # View recent training logs
./train_runpod.sh cleanup        # Clean up old files
./train_runpod.sh package        # Package trained model for download
```

## 📊 Available Presets

### 🐍 qwen-python (Recommended)
- **Best for:** Python code generation and assistance
- **Model:** Qwen2.5-8B (8 billion parameters)
- **Dataset:** 18k Python code instructions
- **Memory:** Optimized for 24GB+ VRAM
- **Training time:** ~2-3 hours on A40

### 💬 qwen-general
- **Best for:** General conversation and instruction following
- **Model:** Qwen2.5-8B
- **Dataset:** Alpaca instruction dataset
- **Memory:** Optimized for 24GB+ VRAM
- **Training time:** ~3-4 hours on A40

### 🦙 llama-python
- **Best for:** Python coding with Llama architecture
- **Model:** Llama 3.1 8B
- **Dataset:** 18k Python code instructions
- **Memory:** Optimized for 24GB+ VRAM
- **Training time:** ~2-3 hours on A40

### ⚡ mistral-code
- **Best for:** Multi-language code generation
- **Model:** Mistral 7B
- **Dataset:** GitHub code repository
- **Memory:** Optimized for 16GB+ VRAM
- **Training time:** ~2-3 hours on A40

## 🚀 Multi-GPU Training

### Benefits of Multi-GPU Training
- **2-4x faster training** compared to single GPU
- **Larger effective batch sizes** for better gradient stability
- **Longer sequence lengths** with combined memory
- **Better model quality** with improved training dynamics

### Multi-GPU Setup
The system automatically detects your GPUs and provides recommendations:

```bash
# Check available GPUs
./train_multi_gpu.sh detect

# Get recommendations based on your hardware
./train_multi_gpu.sh recommend

# Start multi-GPU training
./train_multi_gpu.sh qwen-python-multi  # 2 GPUs
./train_multi_gpu.sh qwen-python-4gpu   # 4 GPUs
```

### Custom Multi-GPU Configuration
```bash
# Train with specific GPUs
./train_multi_gpu.sh custom 0,1 --preset qwen-python

# Interactive multi-GPU setup
./train_multi_gpu.sh custom 0,1,2,3 --interactive

# Advanced multi-GPU training
./train_multi_gpu.sh custom 0,1 \
  --model_name "Qwen/Qwen2.5-8B" \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2
```

## 🔧 Memory Optimization

The system automatically detects your GPU memory and recommends appropriate settings:

### Single GPU
- **16GB VRAM:** Use `qwen-small` preset
- **24GB VRAM:** Use standard presets like `qwen-python`
- **48GB+ VRAM:** Use `qwen-large` preset for best quality

### Multi-GPU
- **2x 24GB GPUs:** Use `qwen-python-multi` (48GB total)
- **4x 24GB GPUs:** Use `qwen-python-4gpu` (96GB total)
- **Custom:** Use `./train_multi_gpu.sh custom` with your GPU IDs

## 🎛️ Custom Configuration

### Interactive Mode
```bash
./train_runpod.sh interactive
```
This will guide you through:
- Model selection
- Dataset selection
- Training parameters
- LoRA settings
- Quantization options

### Configuration Management
```bash
# Save a configuration
python3 config_manager.py save my_config config.json

# Load a saved configuration
./train_runpod.sh config my_config.json

# Compare configurations
python3 config_manager.py compare config1 config2
```

## 📈 Monitoring Training

### Real-time Monitoring
```bash
./train_runpod.sh monitor
```
Shows:
- GPU utilization and memory
- CPU and RAM usage
- Training progress
- Temperature monitoring

### View Logs
```bash
./train_runpod.sh logs
```
Shows the most recent training logs with key metrics.

## 📦 After Training

### Package Your Model
```bash
./train_runpod.sh package
```
This creates a compressed archive of your trained model that you can download using `runpodctl send`.

### Transfer Files
```bash
# Send model to another RunPod instance
runpodctl send trained_model_TIMESTAMP.tar.gz

# Or download to local machine
runpodctl receive YOUR_TRANSFER_ID
```

## 🛠️ Troubleshooting

### Out of Memory
- Use `./train_runpod.sh qwen-small` for lower memory usage
- Check GPU memory with `./train_runpod.sh gpu-check`
- Reduce batch size in interactive mode

### Slow Training
- Use larger batch sizes if you have memory
- Enable gradient checkpointing
- Use 4-bit quantization

### Model Quality Issues
- Increase LoRA rank (32 → 64)
- Train for more epochs
- Use higher learning rate
- Try different datasets

## 📚 Advanced Usage

### Direct Training Script
```bash
python3 qwen3_training.py \
    --model_name "Qwen/Qwen2.5-8B" \
    --dataset_name "iamtarun/python_code_instructions_18k_alpaca" \
    --output_dir "./my-model" \
    --num_train_epochs 3 \
    --quantization 4bit
```

### Configuration Files
```bash
# Save current preset as config
./train_runpod.sh qwen-python --save-config my_config.json

# Use saved config
python3 train_cli.py --config my_config.json
```

## 🎯 Best Practices for RunPod

1. **Always check GPU memory first:** `./train_runpod.sh gpu-check`
2. **Start with presets:** They're optimized for RunPod hardware
3. **Monitor training:** Use `./train_runpod.sh monitor` in a separate terminal
4. **Package models:** Use `./train_runpod.sh package` before instance termination
5. **Save configs:** Save successful configurations for future use

## 🆘 Support

For issues or questions:
1. Check `./train_runpod.sh help` for comprehensive documentation
2. Use `python3 training_help.py troubleshooting` for common issues
3. Review logs with `./train_runpod.sh logs`

Happy training! 🚀
EOF

echo "📝 Creating quick start script..."
# Create a quick start script
cat > "$TEMP_DIR/QUICK_START.sh" << 'EOF'
#!/bin/bash

echo "🚀 AI Training CLI - Quick Start"
echo "==============================="
echo ""
echo "This will set up everything and start training a Python code assistant."
echo ""
read -p "Continue? [y/N]: " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo "🔧 Running setup..."
    chmod +x runpod_setup_script.sh
    ./runpod_setup_script.sh
else
    echo "❌ Setup cancelled. Run './runpod_setup_script.sh' when ready."
fi
EOF

chmod +x "$TEMP_DIR/QUICK_START.sh"

echo "📋 Creating file list..."
# Create a file list
cat > "$TEMP_DIR/FILES.txt" << 'EOF'
AI Training CLI System Files:

Core Training:
- qwen3_training.py          # Main training script with CLI support
- requirements.txt           # Python dependencies
- test_setup.py             # Setup verification
- runpod_setup_script.sh    # Complete RunPod setup

CLI System:
- train_cli.py              # User-friendly training interface
- config_manager.py         # Configuration management
- training_help.py          # Comprehensive help system
- train.sh                  # Local CLI launcher

RunPod Specific:
- train_runpod.sh           # RunPod-optimized CLI (created by setup)
- RUNPOD_README.md          # This documentation
- QUICK_START.sh            # One-click setup
- FILES.txt                 # This file list

Generated During Setup:
- monitor_training.py       # System monitoring
- run_training.sh           # Training launcher with logging
- logs/                     # Training logs directory
- configs/                  # Saved configurations
- outputs/                  # Model outputs
EOF

echo "📦 Creating package..."
# Create the package
tar -czf "$PACKAGE_NAME" -C "$TEMP_DIR" .

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo "✅ Package created: $PACKAGE_NAME"
echo ""
echo "📤 To use on RunPod:"
echo "1. Upload $PACKAGE_NAME to your RunPod instance"
echo "2. Extract: tar -xzf $PACKAGE_NAME"
echo "3. Run: ./QUICK_START.sh"
echo ""
echo "📋 Package contents:"
echo "- Complete CLI training system"
echo "- RunPod-optimized setup script"
echo "- Comprehensive documentation"
echo "- Multiple training presets"
echo "- Configuration management tools"
echo "- Real-time monitoring"
echo ""
echo "🎯 The package is ready for RunPod deployment!"
