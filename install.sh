#!/bin/bash

# RunPod AI Training CLI System - Complete Installation Script
# This script automates the entire setup process for RunPod instances

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Function to print section headers
print_header() {
    echo ""
    print_color $CYAN "=================================="
    print_color $CYAN "🎯 $1"
    print_color $CYAN "=================================="
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect GPU information
detect_gpu_info() {
    print_header "GPU Detection"
    
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_color $GREEN "✅ Detected $GPU_COUNT GPU(s)"
        
        # Show GPU details
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r index name memory; do
            print_color $CYAN "   GPU $index: $name ($memory)"
        done
        
        # Get memory info for recommendations
        TOTAL_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        MEMORY_GB=$((TOTAL_MEMORY / 1024))
        
        print_color $BLUE "💾 Memory per GPU: ${MEMORY_GB}GB"
        print_color $BLUE "💾 Total GPU Memory: $((MEMORY_GB * GPU_COUNT))GB"
        
        # Store for later use
        echo "$GPU_COUNT" > .gpu_count
        echo "$MEMORY_GB" > .memory_gb
    else
        print_color $RED "❌ NVIDIA driver not found"
        echo "0" > .gpu_count
        echo "0" > .memory_gb
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_header "System Dependencies"
    
    print_color $YELLOW "📦 Updating system packages..."
    apt-get update -qq
    
    print_color $YELLOW "📦 Installing essential packages..."
    apt-get install -y -qq \
        git \
        wget \
        curl \
        vim \
        htop \
        tmux \
        unzip \
        build-essential \
        python3-dev \
        python3-pip \
        ninja-build
    
    print_color $GREEN "✅ System dependencies installed"
}

# Function to install Python dependencies
install_python_deps() {
    print_header "Python Dependencies"
    
    print_color $YELLOW "🐍 Upgrading pip..."
    python3 -m pip install --upgrade pip
    
    print_color $YELLOW "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    print_color $YELLOW "📚 Installing core ML libraries..."
    pip install transformers>=4.45.0 datasets>=2.14.0 tokenizers>=0.15.0
    pip install trl>=0.11.0 peft>=0.12.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
    pip install tensorboard>=2.14.0 huggingface-hub>=0.19.0
    pip install numpy>=1.24.0 pandas>=2.0.0 psutil>=5.9.0
    
    print_color $YELLOW "⚡ Installing Flash Attention (optional)..."
    pip install flash-attn --no-build-isolation || print_color $YELLOW "⚠️  Flash Attention installation failed - training will use standard attention"
    
    print_color $YELLOW "🔧 Installing additional utilities..."
    pip install GPUtil>=1.4.0 nvidia-ml-py3>=7.352.0
    
    print_color $GREEN "✅ Python dependencies installed"
}

# Function to verify installations
verify_installation() {
    print_header "Installation Verification"
    
    print_color $YELLOW "🧪 Testing PyTorch installation..."
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
    
    print_color $YELLOW "🧪 Testing transformers..."
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    
    print_color $YELLOW "🧪 Testing other libraries..."
    python3 -c "import datasets, peft, trl; print('✅ Core libraries working')"
    
    print_color $GREEN "✅ All installations verified"
}

# Function to setup environment
setup_environment() {
    print_header "Environment Setup"
    
    print_color $YELLOW "📁 Creating directories..."
    mkdir -p logs configs outputs checkpoints model_cache
    
    print_color $YELLOW "🌍 Setting environment variables..."
    # Create environment file
    cat > .env << 'EOF'
# AI Training Environment Variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_DISABLED=true
export HF_HOME=./model_cache
EOF
    
    # Source environment in bashrc if not already there
    if ! grep -q "source.*\.env" ~/.bashrc; then
        echo "source $(pwd)/.env" >> ~/.bashrc
    fi
    
    # Source for current session
    source .env
    
    print_color $YELLOW "🔧 Making scripts executable..."
    chmod +x *.sh *.py 2>/dev/null || true

    print_color $YELLOW "📦 Installing tmux for persistent training..."
    apt-get install -y -qq tmux screen
    
    print_color $GREEN "✅ Environment setup complete"
}

# Function to create monitoring script
create_monitoring() {
    print_header "Monitoring Setup"
    
    print_color $YELLOW "📊 Creating enhanced monitoring script..."
    cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced training monitor for RunPod
"""
import time
import psutil
import subprocess
import os
import json
from datetime import datetime

def get_gpu_info():
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "GPU info unavailable"

def get_process_info():
    try:
        # Find Python training processes
        python_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower() and any('train' in arg for arg in proc.info['cmdline']):
                    python_procs.append(proc.info)
            except:
                continue
        return python_procs
    except:
        return []

def monitor_system():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 80)
    print(f"🕐 Time: {timestamp}")
    
    # System info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"💻 CPU Usage: {cpu_percent:.1f}%")
    print(f"🧠 RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    print(f"💾 Disk Usage: {disk.percent:.1f}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)")
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info and gpu_info != "GPU info unavailable":
        print("🖥️  GPU Information:")
        for line in gpu_info.split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 6:
                    idx, name, mem_used, mem_total, util, temp = parts[:6]
                    power = parts[6] if len(parts) > 6 else "N/A"
                    print(f"   GPU {idx}: {name}")
                    print(f"     Memory: {mem_used}MB / {mem_total}MB ({float(mem_used)/float(mem_total)*100:.1f}%)")
                    print(f"     Utilization: {util}% | Temperature: {temp}°C | Power: {power}W")
    
    # Training processes
    procs = get_process_info()
    if procs:
        print("🚀 Training Processes:")
        for proc in procs:
            print(f"   PID {proc['pid']}: {proc['name']} (CPU: {proc['cpu_percent']:.1f}%, RAM: {proc['memory_percent']:.1f}%)")
    
    print("=" * 80)

if __name__ == "__main__":
    print("🎯 AI Training Monitor Started")
    print("Press Ctrl+C to stop")
    try:
        while True:
            monitor_system()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
EOF
    
    chmod +x monitor_training.py
    print_color $GREEN "✅ Enhanced monitoring script created"
}

# Function to show recommendations
show_recommendations() {
    print_header "Setup Complete - Recommendations"
    
    # Read GPU info
    GPU_COUNT=$(cat .gpu_count 2>/dev/null || echo "0")
    MEMORY_GB=$(cat .memory_gb 2>/dev/null || echo "0")
    
    print_color $GREEN "🎉 Installation completed successfully!"
    echo ""
    
    print_color $BLUE "📊 Your RunPod Configuration:"
    print_color $CYAN "   GPUs: $GPU_COUNT"
    print_color $CYAN "   Memory per GPU: ${MEMORY_GB}GB"
    print_color $CYAN "   Total GPU Memory: $((MEMORY_GB * GPU_COUNT))GB"
    echo ""
    
    print_color $YELLOW "🚀 Recommended Next Steps:"

    if [ "$GPU_COUNT" -gt 1 ]; then
        print_color $GREEN "✨ Multi-GPU Setup Detected!"
        echo "   1. Check GPU setup: ./train_multi_gpu.sh detect"
        echo "   2. Get recommendations: ./train_multi_gpu.sh recommend"
        echo "   3. Start persistent training: ./train_persistent.sh tmux ./train_multi_gpu.sh qwen-python-multi"
        echo "   4. Or quick persistent: ./train_persistent.sh quick-multi"
    else
        echo "   1. Check GPU setup: ./train_runpod.sh gpu-check"
        echo "   2. List available presets: ./train_runpod.sh presets"
        echo "   3. Start persistent training: ./train_persistent.sh tmux ./train_runpod.sh qwen-python"
        echo "   4. Or quick persistent: ./train_persistent.sh quick-single"
    fi
    
    echo ""
    print_color $YELLOW "📚 Available Commands:"
    echo "   ./train_runpod.sh help          # Single GPU help"
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "   ./train_multi_gpu.sh help       # Multi-GPU help"
    fi
    echo "   ./train_persistent.sh help      # Persistent training help"
    echo "   ./train_auto_restart.sh help    # Auto-restart training help"
    echo "   python3 training_help.py        # Comprehensive guide"
    echo "   python3 demo_cli.py             # Demo the CLI system"
    echo "   ./monitor_training.py           # System monitoring"
    echo ""

    print_color $PURPLE "🔄 Persistence Features:"
    echo "   • Training survives SSH disconnects"
    echo "   • Automatic restart on crashes"
    echo "   • Session management with tmux/screen"
    echo "   • Health monitoring and logging"
    echo ""
    
    print_color $PURPLE "🎯 Memory Recommendations:"
    if [ "$MEMORY_GB" -ge 40 ]; then
        print_color $GREEN "   High Memory GPU (${MEMORY_GB}GB) - Use large presets"
        echo "   • Batch size: 6-8 per GPU"
        echo "   • Sequence length: 2048-4096"
        echo "   • LoRA rank: 64-128"
    elif [ "$MEMORY_GB" -ge 20 ]; then
        print_color $YELLOW "   Medium Memory GPU (${MEMORY_GB}GB) - Use standard presets"
        echo "   • Batch size: 4-6 per GPU"
        echo "   • Sequence length: 1024-2048"
        echo "   • LoRA rank: 32-64"
    else
        print_color $RED "   Low Memory GPU (${MEMORY_GB}GB) - Use small presets"
        echo "   • Batch size: 1-2 per GPU"
        echo "   • Sequence length: 512-1024"
        echo "   • LoRA rank: 16-32"
    fi
    
    echo ""
    print_color $CYAN "🔗 Quick Start (Persistent):"
    if [ "$GPU_COUNT" -gt 1 ]; then
        print_color $CYAN "   source .env && ./train_persistent.sh quick-multi"
    else
        print_color $CYAN "   source .env && ./train_persistent.sh quick-single"
    fi
    print_color $CYAN "   ./train_persistent.sh attach    # Attach to session"
    print_color $CYAN "   ./train_persistent.sh logs      # View logs"
    
    # Clean up temp files
    rm -f .gpu_count .memory_gb
}

# Main installation function
main() {
    print_color $CYAN "🎯 RunPod AI Training CLI System Installer"
    print_color $CYAN "==========================================="
    echo ""
    
    print_color $YELLOW "This script will install everything needed for AI training on RunPod:"
    print_color $YELLOW "• System dependencies"
    print_color $YELLOW "• Python packages (PyTorch, Transformers, etc.)"
    print_color $YELLOW "• Environment setup"
    print_color $YELLOW "• Monitoring tools"
    print_color $YELLOW "• GPU detection and optimization"
    echo ""
    
    read -p "Continue with installation? [Y/n]: " confirm
    if [[ $confirm =~ ^[Nn]$ ]]; then
        print_color $RED "❌ Installation cancelled"
        exit 0
    fi
    
    # Check if running as root (needed for apt-get)
    if [ "$EUID" -ne 0 ]; then
        print_color $RED "❌ This script needs to be run as root for system package installation"
        print_color $YELLOW "Please run: sudo ./install.sh"
        exit 1
    fi
    
    # Start installation
    detect_gpu_info
    install_system_deps
    install_python_deps
    verify_installation
    setup_environment
    create_monitoring
    show_recommendations
    
    print_color $GREEN "🎉 Installation completed successfully!"
    print_color $CYAN "Ready to start training! 🚀"
}

# Run main function
main "$@"
