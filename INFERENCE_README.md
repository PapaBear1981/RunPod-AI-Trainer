# Qwen3-8B Python Coder - Inference Setup

This package contains everything needed to run and test your fine-tuned Qwen3-8B Python coding model.

## 🎯 Overview

Your model has been fine-tuned on 18k Python code instructions and is ready for inference. This setup provides:

- **CLI Interface**: Command-line interaction with the model
- **Web Interface**: Beautiful Gradio web UI for easy testing
- **Code Sandbox**: Secure execution environment to test generated code
- **Comprehensive Testing**: Automated test suite to evaluate model performance
- **Monitoring Tools**: System resource monitoring during inference

## 📁 Package Contents

```
├── inference_script.py          # Main inference class and CLI interface
├── web_interface.py            # Gradio web UI with sandbox integration
├── code_sandbox.py            # Secure code execution sandbox
├── test_suite.py              # Comprehensive test suite with execution testing
├── inference_requirements.txt  # Python dependencies
├── setup_inference.sh         # Environment setup script
├── INFERENCE_README.md        # This file
└── [Your trained model folder] # qwen3-python-coder/
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup_inference.sh

# Run setup (installs dependencies, creates scripts)
./setup_inference.sh
```

### 2. Copy Your Trained Model

Copy the `qwen3-python-coder` folder from your training pod to this directory:

```bash
# The folder should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - training_config.json (optional)
```

### 3. Verify Model

```bash
# Check that the model can be loaded
python verify_model.py
```

### 4. Quick Test

```bash
# Run a quick inference test
python quick_test.py
```

### 5. Choose Your Interface

**CLI Interface (Interactive):**
```bash
./run_cli.sh
# or
python inference_script.py --interactive
```

**Web Interface:**
```bash
./run_web.sh
# or
python web_interface.py
```

**Single Command:**
```bash
python inference_script.py --instruction "Write a function to calculate factorial"
```

## 🌐 Web Interface Features

The Gradio web interface provides:

- **Easy Input**: Text boxes for instructions and optional input
- **Generation Controls**: Adjust temperature, top-p, max tokens
- **Multiple Outputs**: Generate multiple variations
- **Example Prompts**: Pre-built examples to try
- **Copy Code**: Easy copy button for generated code
- **Real-time Info**: Generation time and token statistics

**Access**: Open http://localhost:7860 in your browser

## 🔒 Code Sandbox Features

The integrated sandbox allows you to safely test generated code:

### Security Features
- **Blocked Operations**: File access, network requests, subprocess execution
- **Safe Imports**: Only allows math, datetime, json, collections, etc.
- **Timeout Protection**: 10-second execution limit
- **Memory Limits**: Prevents memory exhaustion
- **Isolated Execution**: Code runs in restricted environment

### Execution Modes
- **In-Process**: Fast execution with Python's exec()
- **Subprocess**: Maximum isolation using separate Python process

### Web Interface Integration
- **Run Code Button**: Execute generated code directly in the UI
- **Test Input**: Provide input data for your code
- **Execution Results**: See output, errors, and execution time
- **Safety Indicators**: Clear feedback on execution success/failure

### CLI Integration
- **Interactive Execution**: Option to run code after generation
- **Multiple Modes**: Choose between in-process or subprocess execution
- **Real-time Feedback**: Immediate results and error reporting

## 🧪 Testing Your Model

### Comprehensive Test Suite

```bash
# Run full test suite (18 test cases across 6 categories)
python test_suite.py
```

**Test Categories:**
- Basic Functions (factorial, prime numbers, string manipulation)
- Data Structures (stack, queue, binary search tree)
- Algorithms (binary search, quicksort, fibonacci)
- Object-Oriented Programming (classes, inheritance)
- Advanced Features (decorators, context managers, generators)
- Practical Applications (email validation, CSV parsing, web scraping)
- Error Handling (exception handling)

**Evaluation Metrics:**
- Syntax validity (can the code be parsed?)
- Keyword matching (contains expected terms?)
- Quality estimation (poor/fair/good/excellent)
- Execution time per generation
- Category-specific performance

### Test Results

The test suite generates:
- Console summary with key metrics
- Detailed JSON report (`test_report.json`)
- Category-wise performance breakdown
- Quality distribution analysis

## 📊 Monitoring

### System Monitoring

```bash
# Monitor GPU, CPU, and memory usage
python monitor_inference.py
```

### Performance Tips

**For Better Performance:**
- Use temperature 0.3-0.5 for more focused code
- Use temperature 0.7-1.0 for more creative solutions
- Adjust max_tokens based on expected code length
- Use multiple outputs to get variations

**Memory Management:**
- The model uses ~16-20GB VRAM with 4-bit quantization
- Suitable for RTX 4090, A40, A6000, or similar GPUs
- For smaller GPUs, reduce batch size or use CPU inference

## 🎛️ Configuration Options

### CLI Arguments

```bash
python inference_script.py \
    --model_path ./qwen3-python-coder \
    --instruction "Your coding task" \
    --max_tokens 512 \
    --temperature 0.7 \
    --interactive
```

### Generation Parameters

- **max_tokens**: 50-1024 (length of generated code)
- **temperature**: 0.1-2.0 (creativity vs focus)
- **top_p**: 0.1-1.0 (nucleus sampling)
- **top_k**: 1-100 (top-k sampling)
- **repetition_penalty**: 1.0-1.5 (avoid repetition)

## 💡 Usage Examples

### Basic Function
```
Instruction: Write a function to calculate the sum of a list
Generated: def sum_list(numbers): return sum(numbers)
```

### Class Implementation
```
Instruction: Create a Calculator class with basic operations
Generated: Complete class with add, subtract, multiply, divide methods
```

### Algorithm Implementation
```
Instruction: Implement binary search algorithm
Input: [1, 3, 5, 7, 9, 11, 13, 15]
Generated: Complete binary search function with proper logic
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Ensure the model folder exists and contains required files
   ls -la qwen3-python-coder/
   python verify_model.py
   ```

2. **CUDA Out of Memory**
   ```bash
   # Use CPU inference or smaller batch size
   export CUDA_VISIBLE_DEVICES=""  # Force CPU
   ```

3. **Slow Generation**
   ```bash
   # Check GPU utilization
   nvidia-smi
   python monitor_inference.py
   ```

4. **Web Interface Not Accessible**
   ```bash
   # Check if port 7860 is available
   netstat -tulpn | grep 7860
   ```

### Performance Optimization

- **Faster Generation**: Lower temperature, smaller max_tokens
- **Better Quality**: Higher temperature, more tokens, multiple outputs
- **Memory Efficient**: Use 4-bit quantization (default)
- **CPU Fallback**: Set `device_map="cpu"` in inference script

## 📈 Expected Performance

Based on the test suite, you should expect:

- **Syntax Validity**: 85-95% of generated code should be syntactically correct
- **Keyword Matching**: 80-90% should contain expected programming concepts
- **Quality Distribution**: 
  - Excellent: 20-30%
  - Good: 40-50%
  - Fair: 20-30%
  - Poor: 5-10%

- **Generation Speed**: 2-5 seconds per response on RTX 4090
- **Memory Usage**: 16-20GB VRAM for inference

## 🎉 Next Steps

1. **Test Thoroughly**: Run the test suite to evaluate performance
2. **Experiment**: Try different prompts and generation parameters
3. **Integrate**: Use the inference script in your own applications
4. **Improve**: Collect feedback and consider additional fine-tuning

## 📚 Additional Resources

- **Model Architecture**: Qwen3-8B with LoRA adapters
- **Training Data**: 18k Python code instructions (Alpaca format)
- **Fine-tuning Method**: QLoRA with rank-32 adapters
- **Base Model**: Qwen/Qwen3-8B from Hugging Face

Your fine-tuned model is ready to help with Python coding tasks! 🐍✨
