# RunPod AI Trainer - Qwen3-8B Python Coder

A comprehensive training and inference setup for fine-tuning Qwen3-8B on Python code instructions using RunPod.

## 🎯 Overview

This repository provides a complete pipeline for:
- **Training**: Fine-tune Qwen3-8B on 18k Python code instructions using QLoRA
- **Inference**: Deploy trained models with CLI and web interfaces
- **Evaluation**: Comprehensive scoring system and code sandbox for testing

### Key Features
- **Model**: Qwen3-8B (latest Qwen family model)
- **Dataset**: 18k Python code instructions in Alpaca format
- **Method**: LoRA/QLoRA for parameter-efficient fine-tuning
- **Hardware**: Optimized for A40 GPU (48GB VRAM)
- **Framework**: HuggingFace Transformers + TRL + PEFT

## 📁 Project Structure

```
├── qwen3_training.py       # Main training script
├── requirements.txt        # Python dependencies
├── setup_environment.sh    # Environment setup script
├── run_training.sh         # Training launcher with monitoring
├── test_setup.py          # Setup verification script
├── monitor_training.py     # System monitoring (auto-generated)
├── logs/                  # Training logs and monitoring
└── outputs/               # Model checkpoints and final model
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make scripts executable
chmod +x setup_environment.sh run_training.sh

# Run setup (installs dependencies, creates directories)
./setup_environment.sh
```

### 2. Verify Setup

```bash
# Test that everything is working
python test_setup.py
```

### 3. Start Training

```bash
# Launch training with monitoring
./run_training.sh
```

## ⚙️ Configuration

The training configuration is defined in `qwen3_training.py` in the `ModelConfig` class:

### Key Parameters

- **Model**: `Qwen/Qwen2.5-8B`
- **Max Sequence Length**: 2048 tokens
- **LoRA Rank**: 64 (higher for better performance)
- **LoRA Alpha**: 128 (2x rank)
- **Batch Size**: 4 per device, 4 gradient accumulation steps (effective batch size: 16)
- **Learning Rate**: 2e-4 with cosine scheduler
- **Epochs**: 3
- **Quantization**: 4-bit NF4 with nested quantization

### Memory Optimization

- 4-bit quantization (QLoRA)
- Gradient checkpointing
- Flash Attention 2
- Paged AdamW 8-bit optimizer
- LoRA targeting all linear layers

## 📊 Monitoring

The training includes comprehensive monitoring:

- **System Resources**: CPU, RAM, GPU usage and temperature
- **Training Metrics**: Loss, learning rate, throughput
- **Logs**: Saved to timestamped directories in `logs/`

Monitor in real-time:
```bash
# Watch system resources
python monitor_training.py

# Follow training logs
tail -f logs/YYYYMMDD_HHMMSS/training.log
```

## 🎛️ Customization

### Adjust for Different Hardware

For **smaller GPUs** (24GB):
```python
# In ModelConfig class
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
lora_r = 32
lora_alpha = 64
```

For **larger GPUs** (80GB+):
```python
# In ModelConfig class
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
lora_r = 128
lora_alpha = 256
```

### Change Dataset

```python
# In ModelConfig class
dataset_name = "your-dataset-name"
max_samples = 5000  # Limit samples if needed
```

### Modify LoRA Targets

```python
# In setup_lora_config function
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention only
# or
target_modules = "all-linear"  # All linear layers (recommended)
```

## 📈 Expected Performance

### Training Time (A40 GPU)
- **Setup**: ~5-10 minutes
- **Training**: ~6-8 hours for 3 epochs
- **Total**: ~8 hours including setup and saving

### Memory Usage
- **Model**: ~16GB (4-bit quantized)
- **Training**: ~35-40GB total
- **Peak**: ~45GB during checkpointing

### Dataset Processing
- **18k samples** at 2048 max length
- **Packing enabled** for efficiency
- **Instruction format** with system/user/assistant roles

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   per_device_train_batch_size = 2
   gradient_accumulation_steps = 8
   ```

2. **Flash Attention Issues**
   ```python
   # Remove flash attention
   attn_implementation="eager"  # Instead of "flash_attention_2"
   ```

3. **Dataset Loading Errors**
   ```bash
   # Check internet connection and HF token
   huggingface-cli login
   ```

4. **Slow Training**
   ```python
   # Enable optimizations
   dataloader_pin_memory = True
   dataloader_num_workers = 4
   ```

### Debug Mode

Run with debug logging:
```bash
export TRANSFORMERS_VERBOSITY=debug
python qwen3_training.py
```

## 📝 Output

After training completes, you'll find:

- **Final Model**: `outputs/qwen3-python-coder/`
- **Checkpoints**: `outputs/qwen3-python-coder/checkpoint-*/`
- **Training Logs**: `logs/YYYYMMDD_HHMMSS/training.log`
- **System Monitoring**: `logs/YYYYMMDD_HHMMSS/system_monitor.log`
- **Config**: `outputs/qwen3-python-coder/training_config.json`

## 🧪 Testing the Model

After training, test your model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-8B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "outputs/qwen3-python-coder")

# Test generation
prompt = "### Instruction:\nWrite a Python function to calculate factorial\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🤝 Contributing

Feel free to:
- Adjust hyperparameters for your use case
- Try different datasets
- Experiment with different LoRA configurations
- Add evaluation metrics

## 📄 License

This project follows the same license as the underlying models and datasets used.
