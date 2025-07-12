#!/bin/bash

# Setup script for Qwen3-8B training environment on RunPod
# This script prepares the environment for fine-tuning

set -e  # Exit on any error

echo "🚀 Setting up Qwen3-8B training environment..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
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
    ninja-build

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
nvidia-smi
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements (skip flash-attn if it fails)
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt || echo "⚠️ Some packages failed to install (likely flash-attn), continuing..."

# Install essential packages individually to ensure they're installed
echo "📦 Installing essential packages..."
pip install transformers>=4.45.0 datasets>=2.14.0 tokenizers>=0.15.0
pip install trl>=0.11.0 peft>=0.12.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
pip install tensorboard>=2.14.0 huggingface-hub>=0.19.0
pip install numpy>=1.24.0 pandas>=2.0.0 psutil>=5.9.0

# Try to install Flash Attention (optional)
echo "⚡ Attempting to install Flash Attention (optional)..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention installation failed - training will use standard attention"

# Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import trl; print(f'TRL version: {trl.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"

# Check GPU memory
echo "💾 GPU Memory Information:"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute Capability: {props.major}.{props.minor}')
else:
    print('No CUDA devices available')
"

# Create directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

# Set up environment variables
echo "🌍 Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true  # Disable wandb by default

# Create a simple monitoring script
cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""
Simple training monitor script
"""
import time
import psutil
import GPUtil
import os

def monitor_system():
    """Monitor system resources during training"""
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent:.1f}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    
    # GPU usage
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  GPU Usage: {gpu.load * 100:.1f}%")
            print(f"  GPU Memory: {gpu.memoryUtil * 100:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB)")
            print(f"  GPU Temp: {gpu.temperature}°C")
    except Exception as e:
        print(f"GPU monitoring error: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    while True:
        monitor_system()
        time.sleep(30)  # Monitor every 30 seconds
EOF

chmod +x monitor_training.py

# Create a training launcher script
cat > run_training.sh << 'EOF'
#!/bin/bash

# Training launcher script with monitoring
set -e

echo "🚀 Starting Qwen3-8B training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create log directory with timestamp
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "📝 Logs will be saved to: $LOG_DIR"

# Start monitoring in background
echo "📊 Starting system monitoring..."
python monitor_training.py > "$LOG_DIR/system_monitor.log" 2>&1 &
MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    kill $MONITOR_PID 2>/dev/null || true
    echo "✅ Training session ended."
}
trap cleanup EXIT

# Start training
echo "🎯 Starting training process..."
python qwen3_training.py 2>&1 | tee "$LOG_DIR/training.log"

echo "🎉 Training completed successfully!"
EOF

chmod +x run_training.sh

echo "✅ Environment setup completed!"
echo ""
echo "🎯 To start training, run:"
echo "   ./run_training.sh"
echo ""
echo "📊 To monitor system resources separately:"
echo "   python monitor_training.py"
echo ""
echo "📁 Directory structure:"
echo "   - qwen3_training.py     : Main training script"
echo "   - requirements.txt      : Python dependencies"
echo "   - run_training.sh       : Training launcher with monitoring"
echo "   - monitor_training.py   : System monitoring script"
echo "   - logs/                 : Training logs and monitoring data"
echo "   - outputs/              : Model outputs and checkpoints"
echo ""
echo "🚀 Ready to train Qwen3-8B on Python code instructions!"
