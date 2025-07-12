# 🔄 Persistent Training Guide for RunPod

This guide shows you how to run AI training that survives SSH disconnections, terminal closures, and network interruptions.

## 🎯 Why Persistence Matters

On RunPod, training can take hours or days. Without persistence:
- **SSH disconnects** stop your training
- **Terminal closure** kills the process
- **Network issues** interrupt training
- **Laptop sleep** breaks the connection

With persistence, training continues running even if you:
- Close your laptop
- Lose internet connection
- Disconnect from SSH
- Restart your local machine

## 🚀 Quick Start - Persistent Training

### Method 1: Tmux (Recommended)
```bash
# Start persistent training
./train_persistent.sh tmux ./train_runpod.sh qwen-python

# Quick commands
./train_persistent.sh quick-single    # Single GPU
./train_persistent.sh quick-multi     # Multi-GPU
```

### Method 2: Auto-Restart (Most Robust)
```bash
# Training with automatic restart on failure
./train_auto_restart.sh --persistent ./train_runpod.sh qwen-python

# Multi-GPU with auto-restart
./train_auto_restart.sh --persistent ./train_multi_gpu.sh qwen-python-multi
```

### Method 3: Simple Background (Basic)
```bash
# Basic background execution
./train_persistent.sh nohup ./train_runpod.sh qwen-python
```

## 🎮 Persistence Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Tmux** | Easy to attach/detach, full terminal | Requires tmux knowledge | Most users |
| **Screen** | Similar to tmux, widely available | Less features than tmux | Alternative to tmux |
| **Nohup** | Simple, no dependencies | No interaction, basic logging | Fire-and-forget |
| **Systemd** | System-level service, auto-start | Requires root, complex | Production setups |
| **Auto-restart** | Handles crashes, health monitoring | More complex | Long training runs |

## 📋 Detailed Usage

### Tmux Method (Recommended)

**Start training:**
```bash
./train_persistent.sh tmux ./train_runpod.sh qwen-python
```

**Manage sessions:**
```bash
./train_persistent.sh list      # List active sessions
./train_persistent.sh attach    # Attach to last session
./train_persistent.sh logs      # View logs
./train_persistent.sh stop      # Stop training
```

**Manual tmux commands:**
```bash
tmux list-sessions              # List all sessions
tmux attach-session -t SESSION # Attach to specific session
tmux detach                     # Detach (Ctrl+B then D)
tmux kill-session -t SESSION   # Kill specific session
```

### Auto-Restart Method (Most Robust)

**Basic auto-restart:**
```bash
./train_auto_restart.sh ./train_runpod.sh qwen-python
```

**Advanced options:**
```bash
# Persistent with 5 retries
./train_auto_restart.sh --persistent --max-retries 5 ./train_runpod.sh qwen-python

# Custom retry delay
./train_auto_restart.sh --retry-delay 120 ./train_multi_gpu.sh qwen-python-multi

# Health monitoring every 10 minutes
./train_auto_restart.sh --health-check 600 ./train_cli.py --preset qwen-python
```

**Monitor auto-restart:**
```bash
./train_auto_restart.sh status  # Check status
./train_auto_restart.sh logs    # View logs
./train_auto_restart.sh stop    # Stop auto-restart
```

### Screen Method

**Start with screen:**
```bash
./train_persistent.sh screen ./train_runpod.sh qwen-python
```

**Screen commands:**
```bash
screen -list                    # List sessions
screen -r SESSION              # Attach to session
screen -d SESSION              # Detach session
# Ctrl+A then D to detach from inside
```

### Nohup Method (Simple)

**Start with nohup:**
```bash
./train_persistent.sh nohup ./train_runpod.sh qwen-python
```

**Monitor nohup:**
```bash
tail -f logs/training_*.log     # View logs
ps aux | grep python           # Find process
kill PID                       # Stop training
```

## 🔧 Session Management

### Check What's Running
```bash
# List all persistent sessions
./train_persistent.sh list

# Check auto-restart status
./train_auto_restart.sh status

# Manual checks
tmux list-sessions              # Tmux sessions
screen -list                    # Screen sessions
ps aux | grep python           # Python processes
nvidia-smi                      # GPU usage
```

### Attach to Running Training
```bash
# Attach to last session
./train_persistent.sh attach

# Attach to specific tmux session
tmux attach-session -t SESSION_NAME

# Attach to specific screen session
screen -r SESSION_NAME
```

### View Training Progress
```bash
# View logs from persistent script
./train_persistent.sh logs

# View auto-restart logs
./train_auto_restart.sh logs

# View specific log file
tail -f logs/training_TIMESTAMP.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Stop Training
```bash
# Stop via persistent script
./train_persistent.sh stop

# Stop auto-restart
./train_auto_restart.sh stop

# Manual stop
tmux kill-session -t SESSION_NAME
kill PID
```

## 🛠️ Troubleshooting

### Common Issues

**Tmux not found:**
```bash
sudo apt-get install tmux
```

**Screen not found:**
```bash
sudo apt-get install screen
```

**Session not found:**
```bash
# List all sessions
./train_persistent.sh list
tmux list-sessions
screen -list
```

**Training appears stuck:**
```bash
# Check GPU utilization
nvidia-smi

# Check log files
tail -f logs/*.log

# Check process status
ps aux | grep python
```

**Can't attach to session:**
```bash
# Kill and restart
./train_persistent.sh stop
./train_persistent.sh tmux ./train_runpod.sh qwen-python
```

### Recovery Procedures

**If training crashes:**
```bash
# Check what happened
./train_auto_restart.sh logs

# Restart manually
./train_persistent.sh tmux ./train_runpod.sh qwen-python

# Or use auto-restart
./train_auto_restart.sh --persistent ./train_runpod.sh qwen-python
```

**If you lose connection:**
```bash
# Reconnect to RunPod
ssh user@runpod-instance

# Check what's running
./train_persistent.sh list

# Attach to session
./train_persistent.sh attach
```

**If RunPod instance restarts:**
```bash
# Training will be lost unless using systemd
# Restart training
./train_persistent.sh tmux ./train_runpod.sh qwen-python

# Or use auto-restart for future resilience
./train_auto_restart.sh --persistent ./train_runpod.sh qwen-python
```

## 🎯 Best Practices

### For Short Training (< 2 hours)
```bash
# Simple tmux is sufficient
./train_persistent.sh tmux ./train_runpod.sh qwen-python
```

### For Long Training (> 2 hours)
```bash
# Use auto-restart for resilience
./train_auto_restart.sh --persistent --max-retries 3 ./train_runpod.sh qwen-python
```

### For Production Training
```bash
# Use systemd service (requires root)
./train_persistent.sh systemd ./train_runpod.sh qwen-python

# Or auto-restart with health monitoring
./train_auto_restart.sh --persistent --health-check 300 ./train_runpod.sh qwen-python
```

### For Multi-GPU Training
```bash
# Always use persistence for expensive multi-GPU training
./train_persistent.sh tmux ./train_multi_gpu.sh qwen-python-multi

# Or with auto-restart
./train_auto_restart.sh --persistent ./train_multi_gpu.sh qwen-python-multi
```

## 📊 Monitoring and Logging

### Real-time Monitoring
```bash
# In one terminal - attach to training
./train_persistent.sh attach

# In another terminal - monitor system
./monitor_training.py

# Or watch GPU usage
watch -n 1 nvidia-smi
```

### Log Management
```bash
# View current logs
./train_persistent.sh logs

# View specific log file
tail -f logs/training_TIMESTAMP.log

# View auto-restart logs
./train_auto_restart.sh logs

# Archive old logs
mkdir -p logs/archive
mv logs/training_*.log logs/archive/
```

## 🚀 Quick Reference

```bash
# Start persistent training
./train_persistent.sh tmux ./train_runpod.sh qwen-python

# Start with auto-restart
./train_auto_restart.sh --persistent ./train_runpod.sh qwen-python

# Check what's running
./train_persistent.sh list

# Attach to session
./train_persistent.sh attach

# View logs
./train_persistent.sh logs

# Stop training
./train_persistent.sh stop

# Quick single GPU
./train_persistent.sh quick-single

# Quick multi-GPU
./train_persistent.sh quick-multi
```

## 💡 Pro Tips

1. **Always use persistence** for training > 30 minutes
2. **Use auto-restart** for training > 2 hours
3. **Monitor GPU usage** to ensure training is progressing
4. **Save checkpoints frequently** in case of failures
5. **Use tmux** for interactive monitoring
6. **Use nohup** for fire-and-forget training
7. **Check logs regularly** to catch issues early
8. **Test persistence** with short runs first

With these persistence methods, you can confidently run long AI training sessions on RunPod without worrying about connection issues! 🎉
