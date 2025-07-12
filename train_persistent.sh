#!/bin/bash

# Persistent Training Launcher for RunPod
# Ensures training continues even if SSH connection is lost

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

# Function to show help
show_help() {
    print_color $CYAN "🎯 Persistent Training Launcher"
    echo "==============================="
    echo ""
    print_color $GREEN "Usage:"
    echo "  ./train_persistent.sh [METHOD] [TRAINING_COMMAND]"
    echo ""
    print_color $GREEN "Methods:"
    echo "  tmux     - Use tmux session (recommended)"
    echo "  screen   - Use screen session"
    echo "  nohup    - Use nohup (basic)"
    echo "  systemd  - Use systemd service (advanced)"
    echo ""
    print_color $GREEN "Examples:"
    echo "  ./train_persistent.sh tmux ./train_runpod.sh qwen-python"
    echo "  ./train_persistent.sh tmux ./train_multi_gpu.sh qwen-python-multi"
    echo "  ./train_persistent.sh screen ./train_cli.py --preset qwen-python"
    echo "  ./train_persistent.sh nohup ./train_runpod.sh interactive"
    echo ""
    print_color $GREEN "Session Management:"
    echo "  ./train_persistent.sh list      # List active sessions"
    echo "  ./train_persistent.sh attach    # Attach to session"
    echo "  ./train_persistent.sh logs      # View training logs"
    echo "  ./train_persistent.sh stop      # Stop training"
    echo ""
    print_color $YELLOW "Quick Commands:"
    echo "  ./train_persistent.sh quick-single   # Quick single GPU training"
    echo "  ./train_persistent.sh quick-multi    # Quick multi-GPU training"
    echo ""
}

# Function to check if tmux is available
check_tmux() {
    if ! command -v tmux &> /dev/null; then
        print_color $YELLOW "📦 Installing tmux..."
        if [ "$EUID" -eq 0 ]; then
            apt-get update -qq && apt-get install -y tmux
        else
            print_color $RED "❌ tmux not found and cannot install without sudo"
            print_color $YELLOW "💡 Try: sudo apt-get install tmux"
            return 1
        fi
    fi
    return 0
}

# Function to check if screen is available
check_screen() {
    if ! command -v screen &> /dev/null; then
        print_color $YELLOW "📦 Installing screen..."
        if [ "$EUID" -eq 0 ]; then
            apt-get update -qq && apt-get install -y screen
        else
            print_color $RED "❌ screen not found and cannot install without sudo"
            print_color $YELLOW "💡 Try: sudo apt-get install screen"
            return 1
        fi
    fi
    return 0
}

# Function to start training with tmux
start_tmux_training() {
    local cmd="$1"
    local session_name="ai-training-$(date +%Y%m%d-%H%M%S)"
    
    print_color $BLUE "🚀 Starting persistent training with tmux..."
    print_color $CYAN "   Session: $session_name"
    print_color $CYAN "   Command: $cmd"
    
    # Create tmux session
    tmux new-session -d -s "$session_name" -c "$(pwd)"
    
    # Setup environment in tmux
    tmux send-keys -t "$session_name" "source .env 2>/dev/null || true" Enter
    tmux send-keys -t "$session_name" "export TMUX_SESSION=$session_name" Enter
    
    # Start training
    tmux send-keys -t "$session_name" "$cmd" Enter
    
    # Save session info
    echo "$session_name" > .last_training_session
    echo "tmux" > .last_training_method
    
    print_color $GREEN "✅ Training started in tmux session: $session_name"
    print_color $YELLOW "📋 To attach: tmux attach-session -t $session_name"
    print_color $YELLOW "📋 To detach: Ctrl+B then D"
    print_color $YELLOW "📋 To view logs: ./train_persistent.sh logs"
    
    # Ask if user wants to attach
    echo ""
    read -p "Attach to session now? [Y/n]: " attach
    if [[ ! $attach =~ ^[Nn]$ ]]; then
        print_color $BLUE "🔗 Attaching to session..."
        tmux attach-session -t "$session_name"
    fi
}

# Function to start training with screen
start_screen_training() {
    local cmd="$1"
    local session_name="ai-training-$(date +%Y%m%d-%H%M%S)"
    
    print_color $BLUE "🚀 Starting persistent training with screen..."
    print_color $CYAN "   Session: $session_name"
    print_color $CYAN "   Command: $cmd"
    
    # Create screen session
    screen -dmS "$session_name" bash -c "source .env 2>/dev/null || true; $cmd"
    
    # Save session info
    echo "$session_name" > .last_training_session
    echo "screen" > .last_training_method
    
    print_color $GREEN "✅ Training started in screen session: $session_name"
    print_color $YELLOW "📋 To attach: screen -r $session_name"
    print_color $YELLOW "📋 To detach: Ctrl+A then D"
    print_color $YELLOW "📋 To view logs: ./train_persistent.sh logs"
}

# Function to start training with nohup
start_nohup_training() {
    local cmd="$1"
    local log_file="logs/training_$(date +%Y%m%d_%H%M%S).log"
    
    print_color $BLUE "🚀 Starting persistent training with nohup..."
    print_color $CYAN "   Command: $cmd"
    print_color $CYAN "   Log file: $log_file"
    
    # Create log directory
    mkdir -p logs
    
    # Start with nohup
    nohup bash -c "source .env 2>/dev/null || true; $cmd" > "$log_file" 2>&1 &
    local pid=$!
    
    # Save process info
    echo "$pid" > .last_training_pid
    echo "$log_file" > .last_training_log
    echo "nohup" > .last_training_method
    
    print_color $GREEN "✅ Training started with nohup (PID: $pid)"
    print_color $YELLOW "📋 Log file: $log_file"
    print_color $YELLOW "📋 To view logs: tail -f $log_file"
    print_color $YELLOW "📋 To stop: kill $pid"
}

# Function to create systemd service
create_systemd_service() {
    local cmd="$1"
    local service_name="ai-training"
    local service_file="/etc/systemd/system/${service_name}.service"
    
    if [ "$EUID" -ne 0 ]; then
        print_color $RED "❌ Systemd service creation requires root access"
        return 1
    fi
    
    print_color $BLUE "🚀 Creating systemd service..."
    
    cat > "$service_file" << EOF
[Unit]
Description=AI Training Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStartPre=/bin/bash -c 'source .env || true'
ExecStart=/bin/bash -c 'source .env && $cmd'
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$service_name"
    systemctl start "$service_name"
    
    echo "$service_name" > .last_training_service
    echo "systemd" > .last_training_method
    
    print_color $GREEN "✅ Training started as systemd service: $service_name"
    print_color $YELLOW "📋 To check status: systemctl status $service_name"
    print_color $YELLOW "📋 To view logs: journalctl -u $service_name -f"
    print_color $YELLOW "📋 To stop: systemctl stop $service_name"
}

# Function to list active sessions
list_sessions() {
    print_color $CYAN "🔍 Active Training Sessions"
    echo "=========================="
    
    # Check tmux sessions
    if command -v tmux &> /dev/null; then
        print_color $BLUE "📺 Tmux Sessions:"
        tmux list-sessions 2>/dev/null | grep "ai-training" || echo "   No tmux training sessions"
    fi
    
    # Check screen sessions
    if command -v screen &> /dev/null; then
        print_color $BLUE "🖥️  Screen Sessions:"
        screen -list 2>/dev/null | grep "ai-training" || echo "   No screen training sessions"
    fi
    
    # Check nohup processes
    if [ -f .last_training_pid ]; then
        local pid=$(cat .last_training_pid)
        if ps -p "$pid" > /dev/null 2>&1; then
            print_color $BLUE "🔄 Nohup Process:"
            echo "   PID $pid: $(ps -p $pid -o comm= 2>/dev/null || echo 'Unknown')"
        fi
    fi
    
    # Check systemd services
    if [ -f .last_training_service ]; then
        local service=$(cat .last_training_service)
        if systemctl is-active "$service" &>/dev/null; then
            print_color $BLUE "⚙️  Systemd Service:"
            echo "   $service: $(systemctl is-active $service)"
        fi
    fi
}

# Function to attach to last session
attach_session() {
    if [ ! -f .last_training_method ]; then
        print_color $RED "❌ No previous training session found"
        return 1
    fi
    
    local method=$(cat .last_training_method)
    
    case $method in
        tmux)
            local session=$(cat .last_training_session)
            print_color $BLUE "🔗 Attaching to tmux session: $session"
            tmux attach-session -t "$session"
            ;;
        screen)
            local session=$(cat .last_training_session)
            print_color $BLUE "🔗 Attaching to screen session: $session"
            screen -r "$session"
            ;;
        nohup)
            local log_file=$(cat .last_training_log)
            print_color $BLUE "📋 Showing nohup logs: $log_file"
            tail -f "$log_file"
            ;;
        systemd)
            local service=$(cat .last_training_service)
            print_color $BLUE "📋 Showing systemd logs: $service"
            journalctl -u "$service" -f
            ;;
    esac
}

# Function to view logs
view_logs() {
    if [ ! -f .last_training_method ]; then
        print_color $YELLOW "⚠️  No previous training session found, showing recent logs..."
        if [ -d logs ]; then
            local latest_log=$(find logs -name "*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
            if [ -n "$latest_log" ]; then
                tail -f "$latest_log"
            else
                echo "No log files found"
            fi
        fi
        return
    fi
    
    local method=$(cat .last_training_method)
    
    case $method in
        tmux)
            local session=$(cat .last_training_session)
            print_color $BLUE "📋 Tmux session logs for: $session"
            tmux capture-pane -t "$session" -p
            ;;
        screen)
            print_color $YELLOW "⚠️  Screen doesn't support log capture. Use attach instead."
            ;;
        nohup)
            local log_file=$(cat .last_training_log)
            print_color $BLUE "📋 Nohup logs: $log_file"
            tail -f "$log_file"
            ;;
        systemd)
            local service=$(cat .last_training_service)
            print_color $BLUE "📋 Systemd logs: $service"
            journalctl -u "$service" -f
            ;;
    esac
}

# Function to stop training
stop_training() {
    if [ ! -f .last_training_method ]; then
        print_color $RED "❌ No previous training session found"
        return 1
    fi
    
    local method=$(cat .last_training_method)
    
    case $method in
        tmux)
            local session=$(cat .last_training_session)
            print_color $YELLOW "🛑 Stopping tmux session: $session"
            tmux kill-session -t "$session" 2>/dev/null || echo "Session already stopped"
            ;;
        screen)
            local session=$(cat .last_training_session)
            print_color $YELLOW "🛑 Stopping screen session: $session"
            screen -S "$session" -X quit 2>/dev/null || echo "Session already stopped"
            ;;
        nohup)
            local pid=$(cat .last_training_pid)
            print_color $YELLOW "🛑 Stopping nohup process: $pid"
            kill "$pid" 2>/dev/null || echo "Process already stopped"
            ;;
        systemd)
            local service=$(cat .last_training_service)
            print_color $YELLOW "🛑 Stopping systemd service: $service"
            systemctl stop "$service"
            ;;
    esac
    
    print_color $GREEN "✅ Training stopped"
}

# Main function
main() {
    case "${1:-help}" in
        "help"|"-h"|"--help")
            show_help
            ;;
        
        "tmux")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train_persistent.sh tmux <command>"
                exit 1
            fi
            check_tmux && start_tmux_training "${@:2}"
            ;;
        
        "screen")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train_persistent.sh screen <command>"
                exit 1
            fi
            check_screen && start_screen_training "${@:2}"
            ;;
        
        "nohup")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train_persistent.sh nohup <command>"
                exit 1
            fi
            start_nohup_training "${@:2}"
            ;;
        
        "systemd")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train_persistent.sh systemd <command>"
                exit 1
            fi
            create_systemd_service "${@:2}"
            ;;
        
        "list")
            list_sessions
            ;;
        
        "attach")
            attach_session
            ;;
        
        "logs")
            view_logs
            ;;
        
        "stop")
            stop_training
            ;;
        
        "quick-single")
            check_tmux && start_tmux_training "./train_runpod.sh qwen-python"
            ;;
        
        "quick-multi")
            check_tmux && start_tmux_training "./train_multi_gpu.sh qwen-python-multi"
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
