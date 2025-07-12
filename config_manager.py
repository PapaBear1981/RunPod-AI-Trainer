#!/usr/bin/env python3
"""
Configuration Manager for AI Training
Manage, save, load, and compare training configurations
"""

import json
import os
import argparse
from typing import Dict, Any, List
from datetime import datetime
import shutil

CONFIG_DIR = "./configs"
TEMPLATES_DIR = "./config_templates"

def ensure_directories():
    """Ensure config directories exist"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

def list_configs():
    """List all saved configurations"""
    ensure_directories()
    configs = []
    
    if os.path.exists(CONFIG_DIR):
        for file in os.listdir(CONFIG_DIR):
            if file.endswith('.json'):
                configs.append(file[:-5])  # Remove .json extension
    
    if not configs:
        print("📁 No saved configurations found")
        return
    
    print("📋 Saved Configurations:")
    print("=" * 50)
    
    for config_name in sorted(configs):
        config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get file modification time
            mtime = os.path.getmtime(config_path)
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            
            print(f"📦 {config_name}")
            print(f"   Model: {config.get('model_name', 'Unknown')}")
            print(f"   Dataset: {config.get('dataset_name', 'Unknown')}")
            print(f"   Modified: {modified}")
            if 'description' in config:
                print(f"   Description: {config['description']}")
            print()
            
        except Exception as e:
            print(f"❌ Error reading {config_name}: {e}")

def save_config(config_name: str, config_data: Dict[str, Any], description: str = None):
    """Save a configuration"""
    ensure_directories()
    
    # Add metadata
    config_data['_metadata'] = {
        'created': datetime.now().isoformat(),
        'name': config_name
    }
    
    if description:
        config_data['description'] = description
    
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✅ Configuration '{config_name}' saved to {config_path}")

def load_config(config_name: str) -> Dict[str, Any]:
    """Load a configuration"""
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration '{config_name}' not found")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove metadata for training
    if '_metadata' in config:
        del config['_metadata']
    
    return config

def delete_config(config_name: str):
    """Delete a configuration"""
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration '{config_name}' not found")
        return
    
    os.remove(config_path)
    print(f"🗑️  Configuration '{config_name}' deleted")

def copy_config(source: str, target: str):
    """Copy a configuration"""
    source_path = os.path.join(CONFIG_DIR, f"{source}.json")
    target_path = os.path.join(CONFIG_DIR, f"{target}.json")
    
    if not os.path.exists(source_path):
        print(f"❌ Source configuration '{source}' not found")
        return
    
    if os.path.exists(target_path):
        confirm = input(f"⚠️  Target configuration '{target}' exists. Overwrite? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ Copy cancelled")
            return
    
    shutil.copy2(source_path, target_path)
    
    # Update metadata
    with open(target_path, 'r') as f:
        config = json.load(f)
    
    if '_metadata' in config:
        config['_metadata']['created'] = datetime.now().isoformat()
        config['_metadata']['name'] = target
        config['_metadata']['copied_from'] = source
    
    with open(target_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Configuration copied from '{source}' to '{target}'")

def compare_configs(config1: str, config2: str):
    """Compare two configurations"""
    try:
        c1 = load_config(config1)
        c2 = load_config(config2)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"🔍 Comparing '{config1}' vs '{config2}':")
    print("=" * 60)
    
    all_keys = set(c1.keys()) | set(c2.keys())
    
    differences = []
    for key in sorted(all_keys):
        val1 = c1.get(key, "NOT SET")
        val2 = c2.get(key, "NOT SET")
        
        if val1 != val2:
            differences.append((key, val1, val2))
            print(f"🔄 {key:25}: {val1} → {val2}")
        else:
            print(f"✅ {key:25}: {val1}")
    
    if not differences:
        print("\n🎯 Configurations are identical!")
    else:
        print(f"\n📊 Found {len(differences)} differences")

def export_config(config_name: str, output_file: str):
    """Export configuration to a specific file"""
    try:
        config = load_config(config_name)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📤 Configuration '{config_name}' exported to {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

def import_config(config_name: str, input_file: str):
    """Import configuration from a file"""
    if not os.path.exists(input_file):
        print(f"❌ Input file '{input_file}' not found")
        return
    
    try:
        with open(input_file, 'r') as f:
            config = json.load(f)
        
        save_config(config_name, config, f"Imported from {input_file}")
        print(f"📥 Configuration imported as '{config_name}'")
    except Exception as e:
        print(f"❌ Import failed: {e}")

def create_template_configs():
    """Create template configurations"""
    ensure_directories()
    
    templates = {
        "qwen_python_small": {
            "description": "Small Qwen model for Python coding (low VRAM)",
            "model_name": "Qwen/Qwen2.5-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./qwen-python-small",
            "max_seq_length": 512,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "quantization": "4bit"
        },
        "qwen_python_large": {
            "description": "Large Qwen model for Python coding (high VRAM)",
            "model_name": "Qwen/Qwen2.5-8B",
            "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
            "output_dir": "./qwen-python-large",
            "max_seq_length": 2048,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lora_r": 64,
            "lora_alpha": 128,
            "quantization": "4bit"
        },
        "llama_general": {
            "description": "Llama 3.1 for general instruction following",
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "dataset_name": "tatsu-lab/alpaca",
            "output_dir": "./llama-general",
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "lora_r": 32,
            "lora_alpha": 64,
            "quantization": "4bit"
        }
    }
    
    for name, config in templates.items():
        template_path = os.path.join(TEMPLATES_DIR, f"{name}.json")
        with open(template_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"📝 Created {len(templates)} template configurations in {TEMPLATES_DIR}")

def main():
    parser = argparse.ArgumentParser(
        description="Configuration Manager for AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all saved configurations')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save a configuration')
    save_parser.add_argument('name', help='Configuration name')
    save_parser.add_argument('file', help='JSON file to save')
    save_parser.add_argument('--description', help='Configuration description')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a configuration')
    load_parser.add_argument('name', help='Configuration name')
    load_parser.add_argument('--output', help='Output file (default: stdout)')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a configuration')
    delete_parser.add_argument('name', help='Configuration name')
    
    # Copy command
    copy_parser = subparsers.add_parser('copy', help='Copy a configuration')
    copy_parser.add_argument('source', help='Source configuration name')
    copy_parser.add_argument('target', help='Target configuration name')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two configurations')
    compare_parser.add_argument('config1', help='First configuration')
    compare_parser.add_argument('config2', help='Second configuration')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to file')
    export_parser.add_argument('name', help='Configuration name')
    export_parser.add_argument('output', help='Output file')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration from file')
    import_parser.add_argument('name', help='Configuration name')
    import_parser.add_argument('input', help='Input file')
    
    # Templates command
    subparsers.add_parser('templates', help='Create template configurations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_configs()
    elif args.command == 'save':
        if not os.path.exists(args.file):
            print(f"❌ File '{args.file}' not found")
            return
        with open(args.file, 'r') as f:
            config = json.load(f)
        save_config(args.name, config, args.description)
    elif args.command == 'load':
        try:
            config = load_config(args.name)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"📤 Configuration loaded to {args.output}")
            else:
                print(json.dumps(config, indent=2))
        except Exception as e:
            print(f"❌ {e}")
    elif args.command == 'delete':
        delete_config(args.name)
    elif args.command == 'copy':
        copy_config(args.source, args.target)
    elif args.command == 'compare':
        compare_configs(args.config1, args.config2)
    elif args.command == 'export':
        export_config(args.name, args.output)
    elif args.command == 'import':
        import_config(args.name, args.input)
    elif args.command == 'templates':
        create_template_configs()

if __name__ == "__main__":
    main()
