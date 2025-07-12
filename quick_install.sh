#!/bin/bash

# Quick Install Script for RunPod AI Training CLI
# Minimal setup for users who want to get started immediately

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    echo -e "${1}${2}${NC}"
}

print_color $CYAN "🚀 Quick Install for RunPod AI Training CLI"
print_color $CYAN "=========================================="
echo ""

print_color $YELLOW "This will quickly install everything needed for AI training."
print_color $YELLOW "For detailed installation, use: sudo ./install.sh"
echo ""

# Check if we need sudo
if [ "$EUID" -ne 0 ]; then
    print_color $YELLOW "⚠️  Running without sudo - will install Python packages only"
    SUDO_NEEDED=true
else
    SUDO_NEEDED=false
fi

# Install system packages if we have sudo
if [ "$SUDO_NEEDED" = false ]; then
    print_color $BLUE "📦 Installing system packages..."
    apt-get update -qq
    apt-get install -y -qq git wget curl vim htop tmux unzip build-essential python3-dev python3-pip ninja-build
fi

# Install Python packages
print_color $BLUE "🐍 Installing Python packages..."
python3 -m pip install --upgrade pip

print_color $BLUE "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print_color $BLUE "📚 Installing ML libraries..."
pip install transformers>=4.45.0 datasets>=2.14.0 tokenizers>=0.15.0
pip install trl>=0.11.0 peft>=0.12.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
pip install tensorboard>=2.14.0 huggingface-hub>=0.19.0
pip install numpy>=1.24.0 pandas>=2.0.0 psutil>=5.9.0 GPUtil>=1.4.0

print_color $BLUE "⚡ Installing Flash Attention..."
pip install flash-attn --no-build-isolation || print_color $YELLOW "⚠️  Flash Attention failed - will use standard attention"

# Setup environment
print_color $BLUE "🔧 Setting up environment..."
mkdir -p logs configs outputs checkpoints model_cache
chmod +x *.sh *.py 2>/dev/null || true

# Create environment file
cat > .env << 'EOF'
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_DISABLED=true
export HF_HOME=./model_cache
EOF

# Test installation
print_color $BLUE "🧪 Testing installation..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"

# Show next steps
print_color $GREEN "🎉 Quick installation complete!"
echo ""
print_color $CYAN "🚀 Quick Start Commands:"
print_color $CYAN "   source .env"

# Detect GPUs and show appropriate commands
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        print_color $CYAN "   ./train_multi_gpu.sh qwen-python-multi  # Multi-GPU training"
    else
        print_color $CYAN "   ./train_runpod.sh qwen-python           # Single GPU training"
    fi
else
    print_color $CYAN "   ./train_runpod.sh qwen-python           # Start training"
fi

print_color $CYAN "   ./train_runpod.sh help                  # See all options"
echo ""
print_color $YELLOW "💡 For full installation with monitoring: sudo ./install.sh"
