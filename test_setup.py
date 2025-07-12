#!/usr/bin/env python3
"""
Test script to verify the training setup is working correctly
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuda():
    """Test CUDA availability and GPU memory"""
    logger.info("Testing CUDA setup...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is not available!")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"✅ CUDA available with {device_count} device(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        if memory_gb < 40:  # A40 should have ~48GB
            logger.warning(f"⚠️  GPU {i} has less than 40GB memory. Training may be limited.")
    
    return True

def test_libraries():
    """Test that all required libraries are installed"""
    logger.info("Testing library imports...")
    
    try:
        import transformers
        import datasets
        import trl
        import peft
        import bitsandbytes
        logger.info("✅ All core libraries imported successfully")
        
        logger.info(f"  - transformers: {transformers.__version__}")
        logger.info(f"  - datasets: {datasets.__version__}")
        logger.info(f"  - trl: {trl.__version__}")
        logger.info(f"  - peft: {peft.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Library import failed: {e}")
        return False

def test_flash_attention():
    """Test Flash Attention availability"""
    logger.info("Testing Flash Attention...")
    
    try:
        import flash_attn
        logger.info("✅ Flash Attention available")
        return True
    except ImportError:
        logger.warning("⚠️  Flash Attention not available (optional)")
        return True

def test_model_loading():
    """Test loading a small model to verify setup"""
    logger.info("Testing model loading (using small model for speed)...")
    
    try:
        # Use a small model for testing
        model_name = "Qwen/Qwen2.5-0.5B"
        
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info(f"✅ Model loaded successfully. Parameters: {model.num_parameters():,}")
        
        # Test a simple generation
        test_prompt = "def fibonacci(n):"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Generation test successful:")
        logger.info(f"  Input: {test_prompt}")
        logger.info(f"  Output: {generated_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_dataset_loading():
    """Test loading the training dataset"""
    logger.info("Testing dataset loading...")
    
    try:
        dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
        logger.info(f"Loading dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name, split="train[:10]")  # Load only 10 samples for testing
        logger.info(f"✅ Dataset loaded successfully. Sample count: {len(dataset)}")
        
        # Show a sample
        sample = dataset[0]
        logger.info("Sample data structure:")
        for key, value in sample.items():
            logger.info(f"  {key}: {str(value)[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False

def test_quantization():
    """Test 4-bit quantization setup"""
    logger.info("Testing 4-bit quantization...")
    
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        logger.info("✅ Quantization config created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantization test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Running setup verification tests...")
    logger.info("=" * 60)
    
    tests = [
        ("CUDA Setup", test_cuda),
        ("Library Imports", test_libraries),
        ("Flash Attention", test_flash_attention),
        ("Quantization", test_quantization),
        ("Dataset Loading", test_dataset_loading),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Ready for training.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
