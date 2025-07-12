#!/bin/bash

# Complete setup script for RunPod - copy and paste this entire script into the RunPod terminal

set -e
echo "🚀 Starting Qwen3-8B Training Setup on RunPod..."

# Step 1: Receive the training files
echo "📦 Receiving training files..."
runpodctl receive 6781-loyal-fashion-lima-1

# Step 2: Extract the package
echo "📂 Extracting training package..."
tar -xzf qwen3_training_package_4090.tar.gz

# Step 3: Make scripts executable
echo "🔧 Setting up permissions..."
chmod +x setup_environment.sh

# Step 4: Update system and install dependencies
echo "📚 Installing dependencies..."
apt-get update -qq
apt-get install -y -qq git wget curl vim htop tmux unzip build-essential python3-dev ninja-build

# Step 5: Check CUDA
echo "🔍 Checking CUDA..."
nvidia-smi

# Step 6: Upgrade pip and install PyTorch
echo "🔥 Installing PyTorch..."
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 7: Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Step 8: Install Flash Attention
echo "⚡ Installing Flash Attention..."
pip install flash-attn --no-build-isolation

# Step 9: Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Step 10: Check GPU memory
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

# Step 11: Create directories
echo "📁 Creating directories..."
mkdir -p logs checkpoints outputs

# Step 12: Set environment variables
echo "🌍 Setting environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Step 13: Test the setup
echo "🧪 Testing setup..."
python test_setup.py

# Step 14: Create monitoring script
cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3
import time
import psutil
import subprocess
import os

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "GPU info unavailable"

def monitor_system():
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"GPU Info: {gpu_info}")
    
    print("=" * 60)

if __name__ == "__main__":
    while True:
        monitor_system()
        time.sleep(30)
EOF

chmod +x monitor_training.py

# Step 15: Create training launcher
cat > run_training.sh << 'EOF'
#!/bin/bash
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

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 To start training, run:"
echo "   ./run_training.sh"
echo ""
echo "📊 To monitor in another terminal:"
echo "   python monitor_training.py"
echo ""
echo "🚀 Starting training now..."

# Step 16: Start training
./run_training.sh
