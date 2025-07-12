#!/bin/bash

# Multi-GPU Training Launcher for RunPod
# Handles distributed training setup and execution

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

# Function to detect available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_color $GREEN "🔍 Detected $GPU_COUNT GPU(s)"
        
        # Show GPU details
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r index name memory; do
            print_color $CYAN "   GPU $index: $name ($memory)"
        done
        
        return $GPU_COUNT
    else
        print_color $RED "❌ NVIDIA driver not found"
        return 0
    fi
}

# Function to check GPU memory and recommend settings
recommend_settings() {
    local gpu_count=$1
    local total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local memory_gb=$((total_memory / 1024))
    
    print_color $BLUE "💾 GPU Memory Analysis:"
    print_color $CYAN "   Memory per GPU: ${memory_gb}GB"
    print_color $CYAN "   Total GPUs: $gpu_count"
    print_color $CYAN "   Total Memory: $((memory_gb * gpu_count))GB"
    
    echo ""
    print_color $YELLOW "📊 Recommended Settings:"
    
    if [ $memory_gb -ge 40 ]; then
        print_color $GREEN "   High Memory GPUs (40GB+):"
        print_color $GREEN "   • Batch size per GPU: 6-8"
        print_color $GREEN "   • Sequence length: 2048-4096"
        print_color $GREEN "   • LoRA rank: 64-128"
        print_color $GREEN "   • Quantization: 4bit or none"
    elif [ $memory_gb -ge 20 ]; then
        print_color $YELLOW "   Medium Memory GPUs (20-40GB):"
        print_color $YELLOW "   • Batch size per GPU: 4-6"
        print_color $YELLOW "   • Sequence length: 1024-2048"
        print_color $YELLOW "   • LoRA rank: 32-64"
        print_color $YELLOW "   • Quantization: 4bit (recommended)"
    else
        print_color $RED "   Low Memory GPUs (<20GB):"
        print_color $RED "   • Batch size per GPU: 1-2"
        print_color $RED "   • Sequence length: 512-1024"
        print_color $RED "   • LoRA rank: 16-32"
        print_color $RED "   • Quantization: 4bit (required)"
    fi
    
    echo ""
    print_color $PURPLE "🚀 Multi-GPU Benefits:"
    print_color $PURPLE "   • ${gpu_count}x faster training"
    print_color $PURPLE "   • Larger effective batch size"
    print_color $PURPLE "   • Better gradient stability"
    print_color $PURPLE "   • Can use longer sequences"
}

# Function to setup distributed training environment
setup_distributed_env() {
    local gpu_ids=$1
    local backend=${2:-nccl}
    
    print_color $BLUE "🔧 Setting up distributed training environment..."
    
    # Set CUDA devices
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    print_color $CYAN "   CUDA_VISIBLE_DEVICES=$gpu_ids"
    
    # Set distributed training variables
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export NCCL_P2P_DISABLE=1  # Disable P2P for stability
    export NCCL_IB_DISABLE=1   # Disable InfiniBand if not available
    
    # Calculate world size
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
    export WORLD_SIZE=${#GPU_ARRAY[@]}
    
    print_color $CYAN "   World Size: $WORLD_SIZE"
    print_color $CYAN "   Backend: $backend"
    print_color $CYAN "   P2P Disabled for stability"
}

# Function to run multi-GPU training
run_multi_gpu_training() {
    local preset=$1
    local gpu_ids=$2
    local custom_args="${@:3}"
    
    print_color $BLUE "🚀 Starting multi-GPU training..."
    print_color $CYAN "   Preset: $preset"
    print_color $CYAN "   GPUs: $gpu_ids"
    
    # Setup environment
    setup_distributed_env "$gpu_ids"
    
    # Create log directory
    LOG_DIR="logs/multi_gpu_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    print_color $CYAN "   Logs: $LOG_DIR"
    
    # Start monitoring in background
    print_color $YELLOW "📊 Starting multi-GPU monitoring..."
    python3 monitor_training.py > "$LOG_DIR/system_monitor.log" 2>&1 &
    MONITOR_PID=$!
    
    # Cleanup function
    cleanup() {
        print_color $YELLOW "🧹 Cleaning up..."
        kill $MONITOR_PID 2>/dev/null || true
        print_color $GREEN "✅ Multi-GPU training session ended."
    }
    trap cleanup EXIT
    
    # Build training command
    if [ -n "$preset" ]; then
        CMD="python3 train_cli.py --preset $preset --use_multi_gpu --gpu_ids $gpu_ids $custom_args"
    else
        CMD="python3 train_cli.py --use_multi_gpu --gpu_ids $gpu_ids $custom_args"
    fi
    
    print_color $BLUE "🎯 Executing: $CMD"
    echo ""
    
    # Run training with logging
    eval "$CMD" 2>&1 | tee "$LOG_DIR/training.log"
    
    print_color $GREEN "🎉 Multi-GPU training completed!"
    print_color $CYAN "📁 Logs saved to: $LOG_DIR"
}

# Function to show multi-GPU help
show_help() {
    print_color $CYAN "🎯 Multi-GPU Training for RunPod"
    echo "================================="
    echo ""
    print_color $GREEN "Quick Commands:"
    echo "  ./train_multi_gpu.sh detect              # Detect available GPUs"
    echo "  ./train_multi_gpu.sh recommend           # Get recommendations"
    echo "  ./train_multi_gpu.sh qwen-python-multi   # Train with 2 GPUs"
    echo "  ./train_multi_gpu.sh qwen-python-4gpu    # Train with 4 GPUs"
    echo ""
    print_color $GREEN "Custom Training:"
    echo "  ./train_multi_gpu.sh custom 0,1 --preset qwen-python"
    echo "  ./train_multi_gpu.sh custom 0,1,2,3 --interactive"
    echo ""
    print_color $GREEN "Advanced Options:"
    echo "  ./train_multi_gpu.sh custom 0,1 \\"
    echo "    --model_name 'Qwen/Qwen2.5-8B' \\"
    echo "    --per_device_train_batch_size 6 \\"
    echo "    --gradient_accumulation_steps 2"
    echo ""
    print_color $YELLOW "Multi-GPU Presets:"
    echo "  • qwen-python-multi  : 2 GPUs, Python coding"
    echo "  • qwen-python-4gpu   : 4 GPUs, Python coding (high memory)"
    echo "  • llama-python-multi : 2 GPUs, Llama Python coding"
    echo ""
}

# Main function
main() {
    case "${1:-help}" in
        "help"|"-h"|"--help")
            show_help
            ;;
        
        "detect")
            detect_gpus
            ;;
        
        "recommend")
            gpu_count=$(detect_gpus)
            if [ $gpu_count -gt 1 ]; then
                recommend_settings $gpu_count
            else
                print_color $RED "❌ Multi-GPU training requires at least 2 GPUs"
            fi
            ;;
        
        "qwen-python-multi")
            gpu_count=$(detect_gpus)
            if [ $gpu_count -ge 2 ]; then
                run_multi_gpu_training "qwen-python-multi" "0,1"
            else
                print_color $RED "❌ This preset requires at least 2 GPUs"
            fi
            ;;
        
        "qwen-python-4gpu")
            gpu_count=$(detect_gpus)
            if [ $gpu_count -ge 4 ]; then
                run_multi_gpu_training "qwen-python-4gpu" "0,1,2,3"
            else
                print_color $RED "❌ This preset requires at least 4 GPUs"
            fi
            ;;
        
        "llama-python-multi")
            gpu_count=$(detect_gpus)
            if [ $gpu_count -ge 2 ]; then
                run_multi_gpu_training "llama-python-multi" "0,1"
            else
                print_color $RED "❌ This preset requires at least 2 GPUs"
            fi
            ;;
        
        "custom")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train_multi_gpu.sh custom GPU_IDS [additional_args...]"
                print_color $YELLOW "   Example: ./train_multi_gpu.sh custom 0,1,2,3 --preset qwen-python"
                exit 1
            fi
            
            gpu_ids=$2
            shift 2  # Remove 'custom' and gpu_ids from arguments
            run_multi_gpu_training "" "$gpu_ids" "$@"
            ;;
        
        *)
            print_color $RED "❌ Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Check if we're in the right directory
if [ ! -f "train_cli.py" ]; then
    print_color $RED "❌ train_cli.py not found. Please run from the training directory."
    exit 1
fi

main "$@"
