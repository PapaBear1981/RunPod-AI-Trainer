#!/bin/bash

# One-liner setup script for RunPod
# This is the absolute fastest way to get started

set -e

echo "🚀 RunPod AI Training CLI - One-Liner Setup"
echo "==========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train_cli.py" ]; then
    echo "❌ Not in the right directory. Please run from the cloned repository."
    echo ""
    echo "Quick setup:"
    echo "1. git clone https://github.com/PapaBear1981/RunPod-AI-Trainer.git"
    echo "2. cd RunPod-AI-Trainer"
    echo "3. git checkout feature/cli-multi-gpu-system"
    echo "4. ./setup_runpod.sh"
    exit 1
fi

echo "🔍 Checking environment..."

# Check if we have sudo
if [ "$EUID" -eq 0 ]; then
    echo "✅ Running as root - full installation available"
    INSTALL_CMD="./install.sh"
elif sudo -n true 2>/dev/null; then
    echo "✅ Sudo available - full installation available"
    INSTALL_CMD="sudo ./install.sh"
else
    echo "⚠️  No sudo - using quick installation"
    INSTALL_CMD="./quick_install.sh"
fi

# Make scripts executable
chmod +x *.sh *.py 2>/dev/null || true

echo ""
echo "🎯 Choose installation type:"
echo "1) Full installation (recommended) - requires sudo"
echo "2) Quick installation (Python packages only)"
echo "3) Just make scripts executable and show commands"
echo ""
read -p "Enter choice (1-3) [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
            echo "🚀 Running full installation..."
            eval $INSTALL_CMD
        else
            echo "❌ Full installation requires sudo access"
            echo "🔄 Falling back to quick installation..."
            ./quick_install.sh
        fi
        ;;
    2)
        echo "🚀 Running quick installation..."
        ./quick_install.sh
        ;;
    3)
        echo "🔧 Making scripts executable..."
        chmod +x *.sh *.py
        echo "✅ Scripts are now executable"
        echo ""
        echo "📋 Manual setup required:"
        echo "1. Install Python packages: pip install torch transformers datasets trl peft accelerate bitsandbytes"
        echo "2. Create environment: mkdir -p logs configs outputs checkpoints"
        echo "3. Start training: ./train_runpod.sh qwen-python"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 Quick start commands:"

# Detect GPUs and show appropriate commands
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo "0")
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "   ./train_multi_gpu.sh detect              # Check your $GPU_COUNT GPUs"
        echo "   ./train_multi_gpu.sh qwen-python-multi   # Start multi-GPU training"
    else
        echo "   ./train_runpod.sh gpu-check              # Check your GPU"
        echo "   ./train_runpod.sh qwen-python            # Start training"
    fi
else
    echo "   ./train_runpod.sh qwen-python            # Start training"
fi

echo "   ./train_runpod.sh help                   # See all options"
echo "   python3 training_help.py                 # Comprehensive guide"
echo ""
echo "💡 Don't forget to run: source .env"
