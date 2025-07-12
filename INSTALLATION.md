# 🚀 RunPod AI Training CLI - Installation Guide

This guide shows you how to install and set up the AI Training CLI system on RunPod instances.

## 🎯 Quick Start (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/PapaBear1981/RunPod-AI-Trainer.git
cd RunPod-AI-Trainer
git checkout feature/cli-multi-gpu-system
```

### 2. Run Installation
```bash
# Full installation (recommended)
sudo ./install.sh

# OR quick installation (minimal)
./quick_install.sh
```

### 3. Start Training
```bash
source .env
./train_runpod.sh qwen-python
```

That's it! 🎉

## 📋 Installation Options

### Option 1: Full Installation (Recommended)
```bash
sudo ./install.sh
```

**What it does:**
- Installs all system dependencies
- Sets up Python environment with all packages
- Creates monitoring tools
- Detects GPU configuration
- Provides personalized recommendations
- Sets up environment variables
- Creates directory structure

**Requirements:** Root access (sudo)

### Option 2: Quick Installation
```bash
./quick_install.sh
```

**What it does:**
- Installs Python packages only
- Basic environment setup
- Quick GPU detection
- Minimal configuration

**Requirements:** No sudo needed (but limited functionality)

### Option 3: Manual Installation
Follow the manual steps below if you prefer to install components individually.

## 🔧 Manual Installation Steps

### 1. System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y git wget curl vim htop tmux unzip build-essential python3-dev python3-pip ninja-build
```

### 2. Python Environment
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML libraries
pip install transformers>=4.45.0 datasets>=2.14.0 tokenizers>=0.15.0
pip install trl>=0.11.0 peft>=0.12.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
pip install tensorboard>=2.14.0 huggingface-hub>=0.19.0
pip install numpy>=1.24.0 pandas>=2.0.0 psutil>=5.9.0 GPUtil>=1.4.0

# Install Flash Attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

### 3. Environment Setup
```bash
# Create directories
mkdir -p logs configs outputs checkpoints model_cache

# Make scripts executable
chmod +x *.sh *.py

# Create environment file
cat > .env << 'EOF'
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_DISABLED=true
export HF_HOME=./model_cache
EOF

# Source environment
source .env
```

## 🧪 Verify Installation

### Test PyTorch and CUDA
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Test ML Libraries
```bash
python3 -c "import transformers, datasets, peft, trl; print('✅ All libraries working')"
```

### Test CLI System
```bash
./train_runpod.sh help
./train_multi_gpu.sh detect  # If you have multiple GPUs
```

## 🖥️ GPU-Specific Setup

### Single GPU Setup
```bash
# Check GPU
./train_runpod.sh gpu-check

# List presets
./train_runpod.sh presets

# Start training
./train_runpod.sh qwen-python
```

### Multi-GPU Setup
```bash
# Detect GPUs
./train_multi_gpu.sh detect

# Get recommendations
./train_multi_gpu.sh recommend

# Start multi-GPU training
./train_multi_gpu.sh qwen-python-multi
```

## 🛠️ Troubleshooting

### Common Issues

**Permission denied:**
```bash
chmod +x *.sh *.py
```

**CUDA not available:**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Out of memory:**
```bash
# Use smaller presets
./train_runpod.sh qwen-small

# Or adjust manually
./train_runpod.sh interactive
```

**Flash Attention fails:**
```bash
# This is optional - training will work without it
pip install flash-attn --no-build-isolation --force-reinstall
```

### Environment Issues

**Scripts not executable:**
```bash
find . -name "*.sh" -exec chmod +x {} \;
find . -name "*.py" -exec chmod +x {} \;
```

**Environment variables not set:**
```bash
source .env
# Or add to bashrc
echo "source $(pwd)/.env" >> ~/.bashrc
```

## 📊 System Requirements

### Minimum Requirements
- **GPU:** 1x 16GB VRAM (RTX 4090, A40, etc.)
- **RAM:** 32GB system RAM
- **Storage:** 50GB free space
- **CUDA:** 11.8 or 12.1

### Recommended Requirements
- **GPU:** 2x 24GB VRAM (A40, RTX 6000, etc.)
- **RAM:** 64GB system RAM
- **Storage:** 100GB free space (SSD preferred)
- **CUDA:** 12.1

### Optimal Requirements
- **GPU:** 4x 24GB VRAM or higher
- **RAM:** 128GB system RAM
- **Storage:** 200GB NVMe SSD
- **CUDA:** 12.1

## 🎯 RunPod-Specific Notes

### RunPod Templates
The system works with any RunPod template that has:
- NVIDIA drivers installed
- Python 3.8+
- CUDA 11.8 or 12.1

### Popular RunPod Configurations
- **2x A40 (48GB):** Perfect for multi-GPU training
- **4x A40 (96GB):** High-performance training
- **1x A100 (80GB):** Single GPU with large memory
- **8x A40 (192GB):** Maximum performance

### Storage Recommendations
- Use **Network Storage** for datasets and models
- Use **Container Storage** for temporary files
- Mount network storage to `./model_cache` for HuggingFace models

## 🚀 Quick Commands Reference

```bash
# Installation
sudo ./install.sh                    # Full installation
./quick_install.sh                   # Quick installation

# Environment
source .env                          # Load environment

# Single GPU
./train_runpod.sh qwen-python        # Start training
./train_runpod.sh gpu-check          # Check GPU
./train_runpod.sh presets            # List presets

# Multi-GPU
./train_multi_gpu.sh detect          # Detect GPUs
./train_multi_gpu.sh qwen-python-multi  # 2 GPU training
./train_multi_gpu.sh qwen-python-4gpu   # 4 GPU training

# Configuration
./train_runpod.sh interactive        # Interactive setup
python3 config_manager.py list       # Manage configs

# Monitoring
./monitor_training.py                # System monitoring
./train_runpod.sh logs              # View logs

# Help
./train_runpod.sh help              # Single GPU help
./train_multi_gpu.sh help           # Multi-GPU help
python3 training_help.py            # Comprehensive guide
```

## 🎉 Next Steps

After installation:

1. **Test the system:** `./train_runpod.sh gpu-check`
2. **Try a quick training:** `./train_runpod.sh qwen-python`
3. **Explore presets:** `./train_runpod.sh presets`
4. **Read the guides:** `python3 training_help.py`
5. **Configure for your needs:** `./train_runpod.sh interactive`

Happy training! 🚀
