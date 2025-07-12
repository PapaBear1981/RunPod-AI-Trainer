#!/bin/bash

# Setup script for Qwen3-8B Python Coder Inference Pod
# This script prepares the environment for running the trained model

set -e  # Exit on any error

echo "🚀 Setting up Qwen3-8B Python Coder Inference Environment..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    unzip \
    build-essential \
    python3-dev \
    ninja-build

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
nvidia-smi
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r inference_requirements.txt

# Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"

# Check GPU memory
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

# Create directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p model_cache

# Set up environment variables
echo "🌍 Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Create model verification script
cat > verify_model.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that the trained model can be loaded
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def verify_model(model_path="./qwen3-python-coder"):
    """Verify the model can be loaded"""
    print(f"🔍 Verifying model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    try:
        # Check for required files
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                print(f"❌ Missing required file: {file_path}")
                return False
            print(f"✅ Found: {file}")
        
        # Load config
        config = PeftConfig.from_pretrained(model_path)
        print(f"✅ LoRA config loaded: {config.base_model_name_or_path}")
        
        print("✅ Model verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_model()
EOF

chmod +x verify_model.py

# Create quick test script
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test of the inference system
"""
from inference_script import Qwen3PythonCoder

def quick_test():
    """Run a quick test of the model"""
    print("🧪 Running quick inference test...")
    
    try:
        # Initialize model
        coder = Qwen3PythonCoder()
        
        # Test generation
        instruction = "Write a simple function to add two numbers"
        print(f"📝 Test instruction: {instruction}")
        
        responses = coder.generate_code(
            instruction=instruction,
            max_new_tokens=200,
            temperature=0.7
        )
        
        print("🎯 Generated code:")
        print("-" * 40)
        print(responses[0])
        print("-" * 40)
        print("✅ Quick test passed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    quick_test()
EOF

chmod +x quick_test.py

# Create launcher scripts
cat > run_cli.sh << 'EOF'
#!/bin/bash

# Launch CLI interface
echo "🖥️ Starting Qwen3 Python Coder CLI..."
python inference_script.py --interactive
EOF

cat > run_web.sh << 'EOF'
#!/bin/bash

# Launch web interface
echo "🌐 Starting Qwen3 Python Coder Web Interface..."
echo "📱 Access the interface at: http://localhost:7860"
echo "🌍 Or use the public URL if share=True is enabled"
python web_interface.py
EOF

chmod +x run_cli.sh run_web.sh

# Create monitoring script
cat > monitor_inference.py << 'EOF'
#!/usr/bin/env python3
"""
Monitor inference performance
"""
import time
import psutil
import subprocess
import GPUtil

def monitor_system():
    """Monitor system resources during inference"""
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent:.1f}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    
    # GPU usage
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  GPU Usage: {gpu.load * 100:.1f}%")
            print(f"  GPU Memory: {gpu.memoryUtil * 100:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB)")
            print(f"  GPU Temp: {gpu.temperature}°C")
    except Exception as e:
        print(f"GPU monitoring error: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    while True:
        monitor_system()
        time.sleep(10)  # Monitor every 10 seconds
EOF

chmod +x monitor_inference.py

echo "✅ Inference environment setup completed!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy your trained model (qwen3-python-coder folder) to this directory"
echo "2. Run: python verify_model.py"
echo "3. Run: python quick_test.py"
echo "4. Launch interface:"
echo "   - CLI: ./run_cli.sh"
echo "   - Web: ./run_web.sh"
echo ""
echo "📊 Monitor performance:"
echo "   python monitor_inference.py"
echo ""
echo "🚀 Ready for inference!"
