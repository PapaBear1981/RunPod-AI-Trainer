#!/usr/bin/env python3
"""
Fine-tuning Qwen3-8B on Python Code Instructions Dataset
Using LoRA/QLoRA for efficient training on A40 GPU (48GB VRAM)
"""

import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional
import json

# Core ML libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model and training parameters"""
    # Model settings
    model_name: str = "Qwen/Qwen3-8B"  # Using Qwen2.5-8B as it's more recent
    max_seq_length: int = 1024  # Reduced for RTX 4090 memory constraints
    
    # LoRA settings optimized for RTX 4090 (24GB VRAM)
    lora_r: int = 32
    lora_alpha: int = 64  # 2x lora_r for good performance
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set to "all-linear"

    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Training settings
    output_dir: str = "./qwen3-python-coder"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # Conservative for RTX 4090 24GB
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Optimization settings
    optim: str = "paged_adamw_8bit"  # Memory efficient optimizer
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Dataset settings
    dataset_name: str = "iamtarun/python_code_instructions_18k_alpaca"
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # Use all samples
    
    # Misc
    seed: int = 42
    report_to: str = "tensorboard"
    push_to_hub: bool = False

def setup_quantization_config(config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Setup 4-bit quantization configuration for memory efficiency"""
    if not config.use_4bit:
        return None
    
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
        bnb_4bit_quant_storage=compute_dtype,
    )
    
    logger.info(f"Using 4-bit quantization with {config.bnb_4bit_quant_type} and {config.bnb_4bit_compute_dtype}")
    return bnb_config

def setup_lora_config(config: ModelConfig) -> LoraConfig:
    """Setup LoRA configuration for parameter-efficient fine-tuning"""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules="all-linear",  # Target all linear layers
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],  # Save these for better performance
    )
    
    logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    return lora_config

def load_and_prepare_model(config: ModelConfig):
    """Load and prepare the model and tokenizer"""
    logger.info(f"Loading model: {config.model_name}")
    
    # Setup quantization
    bnb_config = setup_quantization_config(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",  # Important for training
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",  # Use Flash Attention for efficiency
        use_cache=False,  # Disable for training
    )
    
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
    return model, tokenizer

def load_and_prepare_dataset(config: ModelConfig):
    """Load and prepare the Python code instructions dataset"""
    logger.info(f"Loading dataset: {config.dataset_name}")
    
    # Load dataset
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    logger.info(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Print a sample to understand the format
    logger.info("Sample from dataset:")
    sample = dataset[0]
    for key, value in sample.items():
        logger.info(f"  {key}: {str(value)[:100]}...")
    
    return dataset

def format_instruction_data(example):
    """Format the dataset into instruction-following format"""
    # The dataset should have 'instruction', 'input', and 'output' fields
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Create a formatted prompt
    if input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}

def main():
    """Main training function"""
    # Initialize configuration
    config = ModelConfig()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    logger.info("Starting Qwen3-8B fine-tuning on Python code instructions")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(config)
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(config)
    
    # Format dataset for instruction following
    formatted_dataset = dataset.map(format_instruction_data, remove_columns=dataset.column_names)
    
    # Setup LoRA
    peft_config = setup_lora_config(config)
    
    # Setup training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=config.dataloader_pin_memory,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        push_to_hub=config.push_to_hub,
        max_seq_length=config.max_seq_length,
        packing=False,  # Enable packing for efficiency
        dataset_text_field="text",
        bf16=True,  # Use bfloat16 for training
        remove_unused_columns=False,
        seed=config.seed,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    logger.info("Starting training...")
    
    # Start training
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"Training completed! Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
