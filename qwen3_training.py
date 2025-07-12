#!/usr/bin/env python3
"""
Fine-tuning Qwen3-8B on Python Code Instructions Dataset
Using LoRA/QLoRA for efficient training on A40 GPU (48GB VRAM)
"""

import os
import torch
import logging
import argparse
from dataclasses import dataclass, asdict
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

    # Multi-GPU settings
    use_multi_gpu: bool = False  # Enable multi-GPU training
    gpu_ids: str = "0"  # Comma-separated GPU IDs (e.g., "0,1,2,3")
    ddp_backend: str = "nccl"  # Distributed backend (nccl for GPU, gloo for CPU)
    ddp_find_unused_parameters: bool = False  # Find unused parameters in DDP
    
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
    import os

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

    # Determine device_map based on distributed training
    device_map = None
    if config.use_multi_gpu and "RANK" in os.environ:
        # For distributed training, don't use device_map="auto"
        device_map = None
        logger.info("🔧 Using distributed training - device_map set to None")
    else:
        # For single GPU or non-distributed training
        device_map = "auto"
        logger.info("🔧 Using single GPU - device_map set to auto")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",  # Use Flash Attention for efficiency
        use_cache=False,  # Disable for training
    )

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Handle multi-GPU with DataParallel for simpler setup
    if config.use_multi_gpu and not ("RANK" in os.environ):
        # Use DataParallel for simpler multi-GPU setup
        gpu_ids = [int(x.strip()) for x in config.gpu_ids.split(',')]
        if len(gpu_ids) > 1 and torch.cuda.device_count() >= len(gpu_ids):
            logger.info(f"🔧 Using DataParallel for multi-GPU training on GPUs: {gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            model = model.cuda()

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

def create_cli_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with LoRA/QLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-8B",
        help="HuggingFace model name or path"
    )
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training"
    )

    # LoRA settings
    lora_group = parser.add_argument_group('LoRA Settings')
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (higher = more parameters, better quality)"
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha (scaling factor, typically 2x lora_r)"
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate"
    )

    # Quantization settings
    quant_group = parser.add_argument_group('Quantization Settings')
    quant_group.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help="Quantization type (none, 4bit, 8bit)"
    )
    quant_group.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Compute dtype for 4-bit quantization"
    )
    quant_group.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        choices=["fp4", "nf4"],
        default="nf4",
        help="4-bit quantization type"
    )

    # Training settings
    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument(
        "--output_dir",
        type=str,
        default="./trained-model",
        help="Output directory for the trained model"
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type"
    )

    # Dataset settings
    data_group = parser.add_argument_group('Dataset Settings')
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
        help="HuggingFace dataset name"
    )
    data_group.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None for all)"
    )

    # Optimization settings
    opt_group = parser.add_argument_group('Optimization Settings')
    opt_group.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer type"
    )
    opt_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing"
    )
    opt_group.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing"
    )

    # Logging and saving
    log_group = parser.add_argument_group('Logging and Saving')
    log_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    log_group.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    log_group.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep"
    )

    # Multi-GPU settings
    gpu_group = parser.add_argument_group('Multi-GPU Settings')
    gpu_group.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Enable multi-GPU training"
    )
    gpu_group.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')"
    )
    gpu_group.add_argument(
        "--ddp_backend",
        type=str,
        choices=["nccl", "gloo"],
        default="nccl",
        help="Distributed backend (nccl for GPU, gloo for CPU)"
    )
    gpu_group.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Find unused parameters in DDP (slower but more robust)"
    )

    # Misc settings
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    misc_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training"
    )
    misc_group.add_argument(
        "--config_file",
        type=str,
        help="Load configuration from JSON file"
    )
    misc_group.add_argument(
        "--save_config",
        type=str,
        help="Save current configuration to JSON file"
    )

    return parser

def load_config_from_file(config_file: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def save_config_to_file(config: ModelConfig, config_file: str):
    """Save configuration to JSON file"""
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Configuration saved to {config_file}")

def setup_multi_gpu_environment(config: ModelConfig):
    """Setup multi-GPU training environment"""
    import os
    import torch.distributed as dist

    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    logger.info(f"🖥️  Setting CUDA_VISIBLE_DEVICES={config.gpu_ids}")

    # Verify GPUs are available
    gpu_ids = [int(x.strip()) for x in config.gpu_ids.split(',')]
    available_gpus = torch.cuda.device_count()

    if available_gpus < len(gpu_ids):
        logger.warning(f"⚠️  Requested {len(gpu_ids)} GPUs but only {available_gpus} available")
        # Adjust to available GPUs
        config.gpu_ids = ','.join([str(i) for i in range(available_gpus)])
        logger.info(f"🔧 Adjusted to use GPUs: {config.gpu_ids}")

    # Set distributed training environment variables
    if len(gpu_ids) > 1:
        os.environ["WORLD_SIZE"] = str(len(gpu_ids))
        os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P for stability
        logger.info(f"🌍 World size set to {len(gpu_ids)}")

        # Initialize distributed training if using torchrun
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            logger.info(f"🔧 Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

            # Set device for this process
            torch.cuda.set_device(local_rank)

            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=config.ddp_backend,
                    rank=rank,
                    world_size=world_size
                )
                logger.info(f"✅ Distributed training initialized with {config.ddp_backend} backend")

def update_config_from_args(config: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    """Update ModelConfig with command line arguments"""
    # Model settings
    config.model_name = args.model_name
    config.max_seq_length = args.max_seq_length

    # LoRA settings
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout

    # Quantization settings
    config.use_4bit = args.quantization == "4bit"
    config.bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype
    config.bnb_4bit_quant_type = args.bnb_4bit_quant_type
    config.use_nested_quant = args.quantization == "4bit"

    # Training settings
    config.output_dir = args.output_dir
    config.num_train_epochs = args.num_train_epochs
    config.per_device_train_batch_size = args.per_device_train_batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.warmup_ratio = args.warmup_ratio
    config.lr_scheduler_type = args.lr_scheduler_type

    # Dataset settings
    config.dataset_name = args.dataset_name
    config.dataset_split = args.dataset_split
    config.max_samples = args.max_samples

    # Optimization settings
    config.optim = args.optim
    config.gradient_checkpointing = args.gradient_checkpointing

    # Logging and saving
    config.logging_steps = args.logging_steps
    config.save_steps = args.save_steps
    config.save_total_limit = args.save_total_limit

    # Multi-GPU settings
    config.use_multi_gpu = args.use_multi_gpu
    config.gpu_ids = args.gpu_ids
    config.ddp_backend = args.ddp_backend
    config.ddp_find_unused_parameters = args.ddp_find_unused_parameters

    # Misc
    config.seed = args.seed
    config.push_to_hub = args.push_to_hub

    return config

def main():
    """Main training function"""
    # Parse command line arguments
    parser = create_cli_parser()
    args = parser.parse_args()

    # Initialize configuration
    config = ModelConfig()

    # Load config from file if specified
    if args.config_file:
        logger.info(f"Loading configuration from {args.config_file}")
        file_config = load_config_from_file(args.config_file)
        # Update config with file values
        for key, value in file_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Update config with command line arguments (these override file config)
    config = update_config_from_args(config, args)

    # Save config if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)
        return

    # Setup multi-GPU environment if requested
    if config.use_multi_gpu:
        setup_multi_gpu_environment(config)

    # Set seed for reproducibility
    set_seed(config.seed)

    logger.info("🚀 Starting fine-tuning with the following configuration:")
    logger.info(f"📦 Model: {config.model_name}")
    logger.info(f"📊 Dataset: {config.dataset_name}")
    logger.info(f"🔧 Quantization: {'4-bit' if config.use_4bit else 'None'}")
    logger.info(f"🎯 LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info(f"📁 Output directory: {config.output_dir}")
    logger.info(f"🔄 Epochs: {config.num_train_epochs}")
    logger.info(f"📏 Batch size: {config.per_device_train_batch_size} (effective: {config.per_device_train_batch_size * config.gradient_accumulation_steps})")

    if config.use_multi_gpu:
        gpu_count = len(config.gpu_ids.split(','))
        effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps * gpu_count
        logger.info(f"🖥️  Multi-GPU: {gpu_count} GPUs ({config.gpu_ids})")
        logger.info(f"📏 Total effective batch size: {effective_batch_size}")
        logger.info(f"🔗 DDP Backend: {config.ddp_backend}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config for reference
    with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)
    
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
        # Multi-GPU settings
        ddp_backend=config.ddp_backend if config.use_multi_gpu else None,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters if config.use_multi_gpu else False,
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
