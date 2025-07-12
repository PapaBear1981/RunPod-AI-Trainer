# 🚀 Complete RunPod Deployment Guide

This is your complete guide to deploying and using the AI Training CLI system on RunPod with full persistence support.

## 🎯 What You Get

✅ **Complete CLI System** - No more editing Python code  
✅ **Multi-GPU Support** - 2-8x faster training  
✅ **Persistence** - Training survives disconnects  
✅ **Auto-Restart** - Handles crashes automatically  
✅ **Easy Installation** - One-command setup  
✅ **Comprehensive Monitoring** - Real-time GPU/CPU tracking  

## 🚀 Quick Deployment (3 Steps)

### Step 1: Clone the Repository
```bash
git clone https://github.com/PapaBear1981/RunPod-AI-Trainer.git
cd RunPod-AI-Trainer
git checkout feature/cli-multi-gpu-system
```

### Step 2: Install Everything
```bash
# Full installation (recommended)
sudo ./install.sh

# OR quick installation (no sudo needed)
./quick_install.sh

# OR minimal setup
./setup_runpod.sh
```

### Step 3: Start Persistent Training
```bash
# Load environment
source .env

# Single GPU persistent training
./train_persistent.sh quick-single

# Multi-GPU persistent training (if you have 2+ GPUs)
./train_persistent.sh quick-multi
```

**That's it!** Your training will continue even if you disconnect from SSH! 🎉

## 🔄 Persistence Features

### Why Persistence Matters
- **SSH disconnects** won't stop training
- **Terminal closure** won't kill processes  
- **Network issues** won't interrupt training
- **Laptop sleep** won't break connections

### Persistence Methods Available

| Method | Command | Best For |
|--------|---------|----------|
| **Quick Single** | `./train_persistent.sh quick-single` | Most single GPU users |
| **Quick Multi** | `./train_persistent.sh quick-multi` | Most multi-GPU users |
| **Tmux** | `./train_persistent.sh tmux <command>` | Interactive monitoring |
| **Auto-restart** | `./train_auto_restart.sh --persistent <command>` | Long training runs |
| **Nohup** | `./train_persistent.sh nohup <command>` | Fire-and-forget |

## 🎮 Essential Commands

### Start Training
```bash
# Quick persistent training
./train_persistent.sh quick-single    # Single GPU
./train_persistent.sh quick-multi     # Multi-GPU

# Custom persistent training
./train_persistent.sh tmux ./train_runpod.sh qwen-python
./train_persistent.sh tmux ./train_multi_gpu.sh qwen-python-multi

# Auto-restart training (handles crashes)
./train_auto_restart.sh --persistent ./train_runpod.sh qwen-python
```

### Manage Sessions
```bash
./train_persistent.sh list      # List active sessions
./train_persistent.sh attach    # Attach to last session
./train_persistent.sh logs      # View training logs
./train_persistent.sh stop      # Stop training
```

### Monitor Training
```bash
./monitor_training.py           # Real-time system monitoring
./train_auto_restart.sh status  # Auto-restart status
watch -n 1 nvidia-smi           # GPU monitoring
```

## 🖥️ Multi-GPU Features

### Detect Your Setup
```bash
./train_multi_gpu.sh detect      # Detect available GPUs
./train_multi_gpu.sh recommend   # Get recommendations
```

### Multi-GPU Training
```bash
# Preset multi-GPU training
./train_persistent.sh tmux ./train_multi_gpu.sh qwen-python-multi  # 2 GPUs
./train_persistent.sh tmux ./train_multi_gpu.sh qwen-python-4gpu   # 4 GPUs

# Custom multi-GPU
./train_persistent.sh tmux ./train_multi_gpu.sh custom 0,1,2,3 --preset qwen-python
```

### Performance Benefits
- **2 GPUs:** ~1.8x faster training
- **4 GPUs:** ~3.5x faster training  
- **8 GPUs:** ~6-7x faster training

## 🔧 Configuration Options

### Interactive Setup
```bash
./train_runpod.sh interactive        # Single GPU interactive
./train_multi_gpu.sh custom 0,1 --interactive  # Multi-GPU interactive
```

### Presets Available
```bash
# Single GPU presets
./train_runpod.sh presets

# Multi-GPU presets
qwen-python-multi    # 2 GPU Python coder
qwen-python-4gpu     # 4 GPU high-performance
llama-python-multi   # 2 GPU Llama coder
```

### Configuration Management
```bash
python3 config_manager.py list                    # List saved configs
python3 config_manager.py save my_config config.json  # Save config
python3 config_manager.py compare config1 config2     # Compare configs
```

## 🛠️ Troubleshooting

### Connection Lost?
```bash
# Reconnect to RunPod and check what's running
./train_persistent.sh list

# Attach to your training session
./train_persistent.sh attach
```

### Training Crashed?
```bash
# Check auto-restart status
./train_auto_restart.sh status

# View logs to see what happened
./train_persistent.sh logs

# Restart training
./train_persistent.sh quick-single
```

### GPU Issues?
```bash
# Check GPU status
nvidia-smi
./train_multi_gpu.sh detect

# Use memory-optimized settings
./train_runpod.sh qwen-small
```

### Installation Issues?
```bash
# Try different installation methods
sudo ./install.sh          # Full installation
./quick_install.sh          # Python packages only
./setup_runpod.sh          # Minimal setup
```

## 📊 Monitoring and Logs

### Real-time Monitoring
```bash
# System monitoring
./monitor_training.py

# GPU monitoring  
watch -n 1 nvidia-smi

# Training logs
./train_persistent.sh logs
tail -f logs/training_*.log
```

### Session Management
```bash
# List all active sessions
./train_persistent.sh list
tmux list-sessions
screen -list

# Attach to specific session
tmux attach-session -t SESSION_NAME
screen -r SESSION_NAME
```

## 🎯 Best Practices

### For Short Training (< 2 hours)
```bash
./train_persistent.sh quick-single
```

### For Long Training (> 2 hours)  
```bash
./train_auto_restart.sh --persistent ./train_runpod.sh qwen-python
```

### For Multi-GPU Training
```bash
./train_persistent.sh quick-multi
# OR
./train_auto_restart.sh --persistent ./train_multi_gpu.sh qwen-python-multi
```

### For Experimentation
```bash
./train_runpod.sh interactive    # Configure interactively
python3 config_manager.py save my_experiment config.json  # Save config
./train_persistent.sh tmux ./train_cli.py --config my_experiment.json
```

## 📚 Documentation

- **INSTALLATION.md** - Detailed installation guide
- **PERSISTENCE_GUIDE.md** - Complete persistence documentation  
- **MULTI_GPU_GUIDE.md** - Multi-GPU training guide
- **CLI_SYSTEM_OVERVIEW.md** - Complete CLI system overview

## 🎉 Success Workflow

1. **Deploy:** Clone repo and run `sudo ./install.sh`
2. **Start:** Run `./train_persistent.sh quick-single`
3. **Disconnect:** Close laptop, lose connection - training continues!
4. **Reconnect:** SSH back in and run `./train_persistent.sh attach`
5. **Monitor:** Use `./train_persistent.sh logs` to check progress
6. **Complete:** Training finishes automatically, model is saved

## 🔗 Quick Reference

```bash
# Installation
git clone https://github.com/PapaBear1981/RunPod-AI-Trainer.git
cd RunPod-AI-Trainer && git checkout feature/cli-multi-gpu-system
sudo ./install.sh

# Start persistent training
source .env
./train_persistent.sh quick-single    # Single GPU
./train_persistent.sh quick-multi     # Multi-GPU

# Manage sessions
./train_persistent.sh list            # List sessions
./train_persistent.sh attach          # Attach to session
./train_persistent.sh logs            # View logs
./train_persistent.sh stop            # Stop training

# Help
./train_persistent.sh help            # Persistence help
./train_runpod.sh help                # Single GPU help
./train_multi_gpu.sh help             # Multi-GPU help
python3 training_help.py              # Comprehensive guide
```

Your AI training is now bulletproof against disconnections and crashes! 🛡️🚀
