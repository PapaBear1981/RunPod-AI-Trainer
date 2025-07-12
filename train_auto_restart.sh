#!/bin/bash

# Auto-Restart Training Script
# Automatically restarts training if it crashes or fails

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    echo -e "${1}${2}${NC}"
}

# Configuration
MAX_RETRIES=3
RETRY_DELAY=60  # seconds
HEALTH_CHECK_INTERVAL=300  # 5 minutes
LOG_DIR="logs/auto_restart"

# Function to show help
show_help() {
    print_color $CYAN "🔄 Auto-Restart Training Script"
    echo "==============================="
    echo ""
    print_color $GREEN "Usage:"
    echo "  ./train_auto_restart.sh [OPTIONS] <training_command>"
    echo ""
    print_color $GREEN "Options:"
    echo "  --max-retries N     Maximum number of restart attempts (default: 3)"
    echo "  --retry-delay N     Delay between retries in seconds (default: 60)"
    echo "  --health-check N    Health check interval in seconds (default: 300)"
    echo "  --persistent        Run in persistent mode (tmux)"
    echo "  --no-restart        Disable auto-restart (just monitor)"
    echo ""
    print_color $GREEN "Examples:"
    echo "  ./train_auto_restart.sh ./train_runpod.sh qwen-python"
    echo "  ./train_auto_restart.sh --persistent ./train_multi_gpu.sh qwen-python-multi"
    echo "  ./train_auto_restart.sh --max-retries 5 ./train_cli.py --preset qwen-python"
    echo ""
    print_color $GREEN "Monitoring:"
    echo "  ./train_auto_restart.sh status     # Check status"
    echo "  ./train_auto_restart.sh logs       # View logs"
    echo "  ./train_auto_restart.sh stop       # Stop auto-restart"
    echo ""
}

# Function to log with timestamp
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_DIR/auto_restart.log"
}

# Function to check if training process is healthy
check_training_health() {
    local pid=$1
    
    # Check if process is still running
    if ! ps -p "$pid" > /dev/null 2>&1; then
        return 1
    fi
    
    # Check GPU utilization (if available)
    if command -v nvidia-smi &> /dev/null; then
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        if [ "$gpu_util" -lt 5 ]; then
            log_message "WARN" "Low GPU utilization: ${gpu_util}%"
        fi
    fi
    
    # Check if log file is being updated (if it exists)
    local latest_log=$(find logs -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$latest_log" ]; then
        local last_modified=$(stat -c %Y "$latest_log" 2>/dev/null || echo 0)
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_modified))
        
        # If log hasn't been updated in 30 minutes, consider it stalled
        if [ $time_diff -gt 1800 ]; then
            log_message "WARN" "Training log not updated for ${time_diff} seconds"
            return 1
        fi
    fi
    
    return 0
}

# Function to start training with monitoring
start_training_with_restart() {
    local cmd="$1"
    local use_persistent="$2"
    local retry_count=0
    
    mkdir -p "$LOG_DIR"
    log_message "INFO" "Starting auto-restart training: $cmd"
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        log_message "INFO" "Training attempt $((retry_count + 1))/$MAX_RETRIES"
        
        # Start training
        if [ "$use_persistent" = "true" ]; then
            # Use tmux for persistence
            local session_name="auto-restart-$(date +%Y%m%d-%H%M%S)"
            tmux new-session -d -s "$session_name" -c "$(pwd)"
            tmux send-keys -t "$session_name" "source .env 2>/dev/null || true" Enter
            tmux send-keys -t "$session_name" "$cmd" Enter
            
            # Get the training process PID (approximate)
            sleep 5
            local training_pid=$(tmux list-panes -t "$session_name" -F "#{pane_pid}")
            echo "$session_name" > "$LOG_DIR/current_session"
        else
            # Direct execution
            $cmd &
            local training_pid=$!
        fi
        
        echo "$training_pid" > "$LOG_DIR/current_pid"
        log_message "INFO" "Training started with PID: $training_pid"
        
        # Monitor training
        local last_health_check=$(date +%s)
        while true; do
            sleep 30
            
            # Check if process is still running
            if ! ps -p "$training_pid" > /dev/null 2>&1; then
                local exit_code=$?
                log_message "ERROR" "Training process died (exit code: $exit_code)"
                break
            fi
            
            # Periodic health check
            local current_time=$(date +%s)
            if [ $((current_time - last_health_check)) -ge $HEALTH_CHECK_INTERVAL ]; then
                if ! check_training_health "$training_pid"; then
                    log_message "WARN" "Training health check failed"
                fi
                last_health_check=$current_time
            fi
            
            # Check for completion (look for success indicators in logs)
            if [ -f "$LOG_DIR/../training.log" ]; then
                if grep -q "Training completed successfully" "$LOG_DIR/../training.log" 2>/dev/null; then
                    log_message "INFO" "Training completed successfully!"
                    return 0
                fi
            fi
        done
        
        # Training failed, increment retry count
        retry_count=$((retry_count + 1))
        
        if [ $retry_count -lt $MAX_RETRIES ]; then
            log_message "INFO" "Waiting $RETRY_DELAY seconds before retry..."
            sleep $RETRY_DELAY
        fi
    done
    
    log_message "ERROR" "Training failed after $MAX_RETRIES attempts"
    return 1
}

# Function to show status
show_status() {
    print_color $CYAN "🔍 Auto-Restart Training Status"
    echo "==============================="
    
    if [ -f "$LOG_DIR/current_pid" ]; then
        local pid=$(cat "$LOG_DIR/current_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_color $GREEN "✅ Training is running (PID: $pid)"
            
            # Show GPU status if available
            if command -v nvidia-smi &> /dev/null; then
                echo ""
                print_color $BLUE "🖥️  GPU Status:"
                nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
            fi
        else
            print_color $RED "❌ Training process not found"
        fi
    else
        print_color $YELLOW "⚠️  No active training session"
    fi
    
    if [ -f "$LOG_DIR/current_session" ]; then
        local session=$(cat "$LOG_DIR/current_session")
        if tmux has-session -t "$session" 2>/dev/null; then
            print_color $GREEN "📺 Tmux session active: $session"
        fi
    fi
    
    # Show recent log entries
    if [ -f "$LOG_DIR/auto_restart.log" ]; then
        echo ""
        print_color $BLUE "📋 Recent Log Entries:"
        tail -10 "$LOG_DIR/auto_restart.log"
    fi
}

# Function to view logs
view_logs() {
    if [ -f "$LOG_DIR/auto_restart.log" ]; then
        print_color $BLUE "📋 Auto-Restart Logs:"
        tail -f "$LOG_DIR/auto_restart.log"
    else
        print_color $YELLOW "⚠️  No auto-restart logs found"
    fi
}

# Function to stop auto-restart
stop_auto_restart() {
    print_color $YELLOW "🛑 Stopping auto-restart training..."
    
    # Stop current process
    if [ -f "$LOG_DIR/current_pid" ]; then
        local pid=$(cat "$LOG_DIR/current_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            log_message "INFO" "Stopped training process (PID: $pid)"
        fi
        rm -f "$LOG_DIR/current_pid"
    fi
    
    # Stop tmux session
    if [ -f "$LOG_DIR/current_session" ]; then
        local session=$(cat "$LOG_DIR/current_session")
        if tmux has-session -t "$session" 2>/dev/null; then
            tmux kill-session -t "$session"
            log_message "INFO" "Stopped tmux session: $session"
        fi
        rm -f "$LOG_DIR/current_session"
    fi
    
    print_color $GREEN "✅ Auto-restart training stopped"
}

# Parse command line arguments
USE_PERSISTENT=false
NO_RESTART=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --retry-delay)
            RETRY_DELAY="$2"
            shift 2
            ;;
        --health-check)
            HEALTH_CHECK_INTERVAL="$2"
            shift 2
            ;;
        --persistent)
            USE_PERSISTENT=true
            shift
            ;;
        --no-restart)
            NO_RESTART=true
            shift
            ;;
        status)
            show_status
            exit 0
            ;;
        logs)
            view_logs
            exit 0
            ;;
        stop)
            stop_auto_restart
            exit 0
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            # Remaining arguments are the training command
            break
            ;;
    esac
done

# Check if we have a command to run
if [ $# -eq 0 ]; then
    print_color $RED "❌ No training command specified"
    show_help
    exit 1
fi

# Check if tmux is available for persistent mode
if [ "$USE_PERSISTENT" = "true" ] && ! command -v tmux &> /dev/null; then
    print_color $RED "❌ tmux not found but persistent mode requested"
    print_color $YELLOW "💡 Install tmux: sudo apt-get install tmux"
    exit 1
fi

# Start training with auto-restart
TRAINING_CMD="$*"
print_color $BLUE "🚀 Starting auto-restart training..."
print_color $CYAN "   Command: $TRAINING_CMD"
print_color $CYAN "   Max retries: $MAX_RETRIES"
print_color $CYAN "   Retry delay: $RETRY_DELAY seconds"
print_color $CYAN "   Persistent: $USE_PERSISTENT"

if [ "$NO_RESTART" = "true" ]; then
    print_color $YELLOW "⚠️  Auto-restart disabled, monitoring only"
    MAX_RETRIES=1
fi

start_training_with_restart "$TRAINING_CMD" "$USE_PERSISTENT"
