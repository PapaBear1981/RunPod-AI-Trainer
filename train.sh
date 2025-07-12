#!/bin/bash

# AI Training CLI Launcher
# Simple wrapper script for the training system

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

# Function to show help
show_help() {
    print_color $CYAN "🎯 AI Training CLI System"
    echo "=========================="
    echo ""
    print_color $GREEN "Quick Commands:"
    echo "  ./train.sh quick          - Quick start guide"
    echo "  ./train.sh presets        - List available presets"
    echo "  ./train.sh interactive    - Interactive training setup"
    echo "  ./train.sh help           - Comprehensive help system"
    echo ""
    print_color $GREEN "Training Commands:"
    echo "  ./train.sh qwen-python    - Train Qwen Python coder"
    echo "  ./train.sh qwen-general   - Train Qwen general assistant"
    echo "  ./train.sh llama-python   - Train Llama Python coder"
    echo "  ./train.sh mistral-code   - Train Mistral code assistant"
    echo ""
    print_color $GREEN "Configuration Commands:"
    echo "  ./train.sh configs        - Manage saved configurations"
    echo "  ./train.sh compare        - Compare configurations"
    echo "  ./train.sh templates      - Create template configs"
    echo ""
    print_color $YELLOW "Examples:"
    echo "  ./train.sh qwen-python                    # Use preset"
    echo "  ./train.sh config my_config.json         # Use saved config"
    echo "  ./train.sh interactive --save my.json    # Save interactive config"
    echo ""
}

# Function to check if Python scripts exist
check_dependencies() {
    local missing=0
    
    if [ ! -f "train_cli.py" ]; then
        print_color $RED "❌ train_cli.py not found"
        missing=1
    fi
    
    if [ ! -f "config_manager.py" ]; then
        print_color $RED "❌ config_manager.py not found"
        missing=1
    fi
    
    if [ ! -f "qwen3_training.py" ]; then
        print_color $RED "❌ qwen3_training.py not found"
        missing=1
    fi
    
    if [ $missing -eq 1 ]; then
        print_color $RED "❌ Missing required files. Please ensure all scripts are in the current directory."
        exit 1
    fi
}

# Main script logic
main() {
    # Check dependencies
    check_dependencies
    
    # Handle no arguments
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # Handle commands
    case "$1" in
        "help"|"-h"|"--help")
            if [ $# -eq 1 ]; then
                python3 training_help.py
            else
                python3 training_help.py "$2"
            fi
            ;;

        "quick")
            python3 training_help.py quick
            ;;

        "presets")
            python3 train_cli.py --list-presets
            ;;

        "datasets")
            python3 train_cli.py --list-datasets
            ;;

        "interactive")
            shift
            python3 train_cli.py --interactive "$@"
            ;;

        "qwen-python"|"qwen-general"|"llama-python"|"mistral-code")
            print_color $BLUE "🚀 Starting training with preset: $1"
            python3 train_cli.py --preset "$1"
            ;;

        "config")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train.sh config <config_file.json>"
                exit 1
            fi
            print_color $BLUE "🚀 Starting training with config: $2"
            python3 train_cli.py --config "$2"
            ;;

        "configs")
            python3 config_manager.py list
            ;;

        "compare")
            if [ $# -lt 3 ]; then
                print_color $RED "❌ Usage: ./train.sh compare <config1> <config2>"
                exit 1
            fi
            python3 config_manager.py compare "$2" "$3"
            ;;

        "templates")
            python3 config_manager.py templates
            ;;

        "save")
            if [ $# -lt 3 ]; then
                print_color $RED "❌ Usage: ./train.sh save <name> <config_file.json>"
                exit 1
            fi
            python3 config_manager.py save "$2" "$3"
            ;;

        "load")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train.sh load <config_name>"
                exit 1
            fi
            python3 config_manager.py load "$2"
            ;;

        "delete")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train.sh delete <config_name>"
                exit 1
            fi
            python3 config_manager.py delete "$2"
            ;;

        "copy")
            if [ $# -lt 3 ]; then
                print_color $RED "❌ Usage: ./train.sh copy <source> <target>"
                exit 1
            fi
            python3 config_manager.py copy "$2" "$3"
            ;;

        "dry-run")
            if [ $# -lt 2 ]; then
                print_color $RED "❌ Usage: ./train.sh dry-run <preset_or_config>"
                exit 1
            fi
            if [ -f "$2" ]; then
                python3 train_cli.py --config "$2" --dry-run
            else
                python3 train_cli.py --preset "$2" --dry-run
            fi
            ;;

        "monitor")
            print_color $BLUE "📊 Starting training monitor..."
            if [ -f "monitor_training.py" ]; then
                python3 monitor_training.py
            else
                print_color $YELLOW "⚠️  monitor_training.py not found. Using basic monitoring..."
                watch -n 5 'nvidia-smi; echo ""; ps aux | grep python'
            fi
            ;;
        
        *)
            print_color $RED "❌ Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
