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

# Step 15: Create CLI training launcher
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

# Start training with CLI system
echo "🎯 Starting training process..."
if [ $# -eq 0 ]; then
    # Default training
    python qwen3_training.py 2>&1 | tee "$LOG_DIR/training.log"
else
    # Use CLI arguments
    python train_cli.py "$@" 2>&1 | tee "$LOG_DIR/training.log"
fi

echo "🎉 Training completed successfully!"
EOF

chmod +x run_training.sh

# Step 16: Create RunPod CLI launcher
cat > train_runpod.sh << 'EOF'
#!/bin/bash

# RunPod AI Training CLI System
# Optimized for RunPod cloud instances

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_color() {
    echo -e "${1}${2}${NC}"
}

# RunPod-specific environment setup
setup_runpod_env() {
    export CUDA_VISIBLE_DEVICES=0
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

    # Create necessary directories
    mkdir -p logs configs outputs checkpoints
}

show_runpod_help() {
    print_color $CYAN "🎯 RunPod AI Training CLI System"
    echo "================================="
    echo ""
    print_color $GREEN "Quick Start (RunPod Optimized):"
    echo "  ./train_runpod.sh quick          - Quick start guide"
    echo "  ./train_runpod.sh gpu-check      - Check GPU and memory"
    echo "  ./train_runpod.sh presets        - List training presets"
    echo "  ./train_runpod.sh interactive    - Interactive setup"
    echo ""
    print_color $GREEN "Single GPU Presets (RunPod A40 Optimized):"
    echo "  ./train_runpod.sh qwen-python    - Qwen Python coder (recommended)"
    echo "  ./train_runpod.sh qwen-general   - Qwen general assistant"
    echo "  ./train_runpod.sh llama-python   - Llama Python coder"
    echo "  ./train_runpod.sh mistral-code   - Mistral code assistant"
    echo ""
    print_color $GREEN "Multi-GPU Presets (2+ GPUs):"
    echo "  ./train_multi_gpu.sh qwen-python-multi  - 2 GPU Python coder"
    echo "  ./train_multi_gpu.sh qwen-python-4gpu   - 4 GPU Python coder"
    echo "  ./train_multi_gpu.sh llama-python-multi - 2 GPU Llama coder"
    echo ""
    print_color $GREEN "Memory-Optimized Presets:"
    echo "  ./train_runpod.sh qwen-small     - Low memory version (16GB)"
    echo "  ./train_runpod.sh qwen-large     - High memory version (48GB+)"
    echo ""
    print_color $YELLOW "Multi-GPU Commands:"
    echo "  ./train_multi_gpu.sh detect      - Detect available GPUs"
    echo "  ./train_multi_gpu.sh recommend   - Get multi-GPU recommendations"
    echo "  ./train_multi_gpu.sh custom 0,1  - Custom multi-GPU training"
    echo ""
    print_color $YELLOW "RunPod Specific Commands:"
    echo "  ./train_runpod.sh monitor        - Start system monitoring"
    echo "  ./train_runpod.sh logs           - View recent training logs"
    echo "  ./train_runpod.sh cleanup        - Clean up old files"
    echo "  ./train_runpod.sh package        - Package trained model"
    echo ""
}

check_gpu() {
    print_color $BLUE "🔍 GPU and Memory Check"
    echo "========================"

    # Check NVIDIA driver
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo ""

        # Get memory in GB
        TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)

        TOTAL_GB=$((TOTAL_MEM / 1024))
        FREE_GB=$((FREE_MEM / 1024))

        print_color $GREEN "💾 GPU Memory: ${TOTAL_GB}GB total, ${FREE_GB}GB free"

        # Recommend settings based on memory
        if [ $TOTAL_GB -ge 40 ]; then
            print_color $GREEN "✅ High memory GPU detected - can use large presets"
            echo "   Recommended: ./train_runpod.sh qwen-large"
        elif [ $TOTAL_GB -ge 20 ]; then
            print_color $YELLOW "⚠️  Medium memory GPU - use standard presets"
            echo "   Recommended: ./train_runpod.sh qwen-python"
        else
            print_color $RED "⚠️  Low memory GPU - use small presets"
            echo "   Recommended: ./train_runpod.sh qwen-small"
        fi
    else
        print_color $RED "❌ NVIDIA driver not found"
    fi
}

view_logs() {
    print_color $BLUE "📋 Recent Training Logs"
    echo "======================="

    if [ -d "logs" ]; then
        LATEST_LOG=$(find logs -name "training.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_LOG" ]; then
            print_color $GREEN "Latest log: $LATEST_LOG"
            echo ""
            tail -50 "$LATEST_LOG"
        else
            print_color $YELLOW "No training logs found"
        fi
    else
        print_color $YELLOW "No logs directory found"
    fi
}

cleanup_files() {
    print_color $BLUE "🧹 Cleaning up old files"
    echo "========================"

    # Remove old logs (keep last 3)
    if [ -d "logs" ]; then
        LOG_COUNT=$(find logs -maxdepth 1 -type d | wc -l)
        if [ $LOG_COUNT -gt 4 ]; then
            print_color $YELLOW "Removing old log directories..."
            find logs -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | head -n -3 | cut -d' ' -f2- | xargs rm -rf
        fi
    fi

    # Remove temporary files
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    print_color $GREEN "✅ Cleanup completed"
}

package_model() {
    print_color $BLUE "📦 Packaging trained model"
    echo "=========================="

    # Find the most recent model directory
    MODEL_DIR=$(find . -maxdepth 1 -type d -name "*-coder" -o -name "*-model" -o -name "*qwen*" | head -1)

    if [ -z "$MODEL_DIR" ]; then
        print_color $RED "❌ No trained model found"
        return 1
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PACKAGE_NAME="trained_model_${TIMESTAMP}.tar.gz"

    print_color $YELLOW "Packaging $MODEL_DIR..."
    tar -czf "$PACKAGE_NAME" "$MODEL_DIR"

    print_color $GREEN "✅ Model packaged as: $PACKAGE_NAME"
    print_color $CYAN "📤 Use 'runpodctl send' to transfer this file"
}

main() {
    # Setup RunPod environment
    setup_runpod_env

    case "${1:-help}" in
        "help"|"-h"|"--help")
            show_runpod_help
            ;;
        "gpu-check"|"gpu")
            check_gpu
            ;;
        "quick")
            python3 training_help.py quick
            ;;
        "presets")
            python3 train_cli.py --list-presets
            ;;
        "interactive")
            python3 train_cli.py --interactive
            ;;
        "qwen-python"|"qwen-general"|"llama-python"|"mistral-code")
            print_color $BLUE "🚀 Starting RunPod training with preset: $1"
            ./run_training.sh --preset "$1"
            ;;
        "qwen-small")
            print_color $BLUE "🚀 Starting memory-optimized training (16GB)"
            python3 train_cli.py --preset qwen-python --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_seq_length 512
            ;;
        "qwen-large")
            print_color $BLUE "🚀 Starting high-memory training (48GB+)"
            python3 train_cli.py --preset qwen-python --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --max_seq_length 2048
            ;;
        "monitor")
            python3 monitor_training.py
            ;;
        "logs")
            view_logs
            ;;
        "cleanup")
            cleanup_files
            ;;
        "package")
            package_model
            ;;
        "multi-gpu"|"multigpu")
            if [ -f "train_multi_gpu.sh" ]; then
                ./train_multi_gpu.sh "${@:2}"
            else
                print_color $RED "❌ Multi-GPU script not found. Run setup first."
            fi
            ;;
        "detect-gpus")
            if [ -f "train_multi_gpu.sh" ]; then
                ./train_multi_gpu.sh detect
            else
                print_color $RED "❌ Multi-GPU script not found. Run setup first."
            fi
            ;;
        *)
            print_color $RED "❌ Unknown command: $1"
            show_runpod_help
            ;;
    esac
}

main "$@"
EOF

chmod +x train_runpod.sh

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 NEW: CLI Training System Available!"
echo "======================================"
echo ""
echo "🚀 Quick Start Options:"
echo "   ./train_runpod.sh gpu-check       # Check your GPU setup"
echo "   ./train_runpod.sh presets         # List available presets"
echo "   ./train_runpod.sh qwen-python     # Start Python coder training"
echo "   ./train_runpod.sh interactive     # Interactive configuration"
echo ""
echo "📊 Monitoring & Management:"
echo "   ./train_runpod.sh monitor         # System monitoring"
echo "   ./train_runpod.sh logs            # View training logs"
echo "   ./train_runpod.sh package         # Package trained model"
echo ""
echo "🔧 Legacy Training (original method):"
echo "   ./run_training.sh                 # Original training script"
echo ""
echo "📚 For help and examples:"
echo "   ./train_runpod.sh help            # Full help system"
echo "   python3 training_help.py          # Comprehensive guide"
echo ""

# Step 17: Show GPU info and recommendations
echo "🔍 Checking your RunPod GPU setup..."
./train_runpod.sh gpu-check

# Check for multi-GPU capability
echo ""
echo "🖥️  Checking multi-GPU capability..."
chmod +x train_multi_gpu.sh
GPU_COUNT=$(./train_multi_gpu.sh detect | grep "Detected" | grep -o '[0-9]\+' | head -1)

if [ "$GPU_COUNT" -gt 1 ]; then
    echo ""
    print_color $GREEN "🚀 Multi-GPU detected! You have $GPU_COUNT GPUs available."
    print_color $CYAN "   Multi-GPU training can be 2-4x faster than single GPU."
    echo ""
    ./train_multi_gpu.sh recommend
fi

echo ""

# Ask user what they want to do
echo "🤔 What would you like to do?"
echo "1) Start training with Qwen Python preset (single GPU)"
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "2) Start multi-GPU training with Qwen Python (${GPU_COUNT} GPUs)"
    echo "3) See all available presets (single and multi-GPU)"
    echo "4) Use interactive configuration"
    echo "5) Just finish setup (train later)"
    echo ""
    read -p "Enter your choice (1-5): " choice
else
    echo "2) See all available presets"
    echo "3) Use interactive configuration"
    echo "4) Just finish setup (train later)"
    echo ""
    read -p "Enter your choice (1-4): " choice
fi

if [ "$GPU_COUNT" -gt 1 ]; then
    case $choice in
        1)
            echo "🚀 Starting single GPU Qwen Python training..."
            ./train_runpod.sh qwen-python
            ;;
        2)
            echo "🚀 Starting multi-GPU Qwen Python training..."
            if [ "$GPU_COUNT" -ge 4 ]; then
                ./train_multi_gpu.sh qwen-python-4gpu
            else
                ./train_multi_gpu.sh qwen-python-multi
            fi
            ;;
        3)
            ./train_runpod.sh presets
            echo ""
            print_color $CYAN "Multi-GPU presets:"
            ./train_multi_gpu.sh help | grep "qwen-python\|llama-python"
            echo ""
            read -p "Enter preset name to use (or press Enter to skip): " preset
            if [ -n "$preset" ]; then
                if [[ "$preset" == *"multi"* ]] || [[ "$preset" == *"4gpu"* ]]; then
                    ./train_multi_gpu.sh "$preset"
                else
                    ./train_runpod.sh "$preset"
                fi
            fi
            ;;
        4)
            ./train_runpod.sh interactive
            ;;
        5)
            echo "✅ Setup complete! Use './train_runpod.sh help' or './train_multi_gpu.sh help' to get started."
            ;;
        *)
            echo "✅ Setup complete! Use './train_runpod.sh help' or './train_multi_gpu.sh help' to get started."
            ;;
    esac
else
    case $choice in
        1)
            echo "🚀 Starting Qwen Python training..."
            ./train_runpod.sh qwen-python
            ;;
        2)
            ./train_runpod.sh presets
            echo ""
            read -p "Enter preset name to use (or press Enter to skip): " preset
            if [ -n "$preset" ]; then
                ./train_runpod.sh "$preset"
            fi
            ;;
        3)
            ./train_runpod.sh interactive
            ;;
        4)
            echo "✅ Setup complete! Use './train_runpod.sh help' to get started."
            ;;
        *)
            echo "✅ Setup complete! Use './train_runpod.sh help' to get started."
            ;;
    esac
fi
