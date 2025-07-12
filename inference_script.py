#!/usr/bin/env python3
"""
Qwen3-8B Python Coder Inference Script
Load and test the fine-tuned LoRA model for Python code generation
"""

import os
import torch
import logging
import argparse
from typing import Optional, List, Dict
import json
import time

# Core ML libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel, PeftConfig

# Import sandbox and scoring system
try:
    from code_sandbox import CodeSandbox
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    logger.warning("Code sandbox not available. Install code_sandbox.py for code execution.")

try:
    from scoring_system import CodeScorer
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False
    logger.warning("Scoring system not available. Install scoring_system.py for code scoring.")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Qwen3PythonCoder:
    """Inference class for the fine-tuned Qwen3-8B Python coding model"""
    
    def __init__(
        self,
        model_path: str = "./qwen3-python-coder",
        base_model: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_memory: Optional[Dict] = None
    ):
        """
        Initialize the model for inference
        
        Args:
            model_path: Path to the trained LoRA adapter
            base_model: Base model identifier
            device: Device to load model on
            load_in_4bit: Whether to use 4-bit quantization
            max_memory: Memory allocation per device
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        
        self.tokenizer = None
        self.model = None
        self.sandbox = CodeSandbox(timeout=10) if SANDBOX_AVAILABLE else None
        self.scorer = CodeScorer() if SCORING_AVAILABLE else None

        logger.info(f"Initializing Qwen3 Python Coder from {model_path}")
        if self.scorer:
            self.scorer.load_scores()
        self._load_model()
    
    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization configuration"""
        if not self.load_in_4bit:
            return None
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        
        logger.info("Using 4-bit quantization for inference")
        return bnb_config
    
    def _load_model(self):
        """Load the base model and LoRA adapter"""
        try:
            # Setup quantization
            bnb_config = self._setup_quantization_config()
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="left",  # For inference
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model
            logger.info(f"Loading base model: {self.base_model}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",  # Avoid Flash Attention issues
                max_memory=self.max_memory,
            )
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Base model parameters: {base_model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the prompt in the same way as training data"""
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt
    
    def generate_code(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate Python code based on instruction
        
        Args:
            instruction: The coding instruction/task
            input_text: Optional input context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated code strings
        """
        # Format prompt
        prompt = self.format_prompt(instruction, input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            start_time = time.time()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            generation_time = time.time() - start_time
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove the input prompt from output
            generated_tokens = output[inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        logger.info(f"Generated {len(generated_texts)} responses in {generation_time:.2f}s")
        return generated_texts
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("🐍 Qwen3 Python Coder - Interactive Mode")
        print("=" * 50)
        print("Enter your Python coding instructions. Type 'quit' to exit.")
        print("Type 'help' for example prompts.")
        print("=" * 50)
        
        while True:
            try:
                instruction = input("\n💻 Instruction: ").strip()
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if instruction.lower() == 'help':
                    self._show_help()
                    continue

                if instruction.lower() in ['scores', 'highscores', 'high-scores']:
                    self._show_high_scores()
                    continue

                if instruction.lower() in ['stats', 'statistics']:
                    self._show_statistics()
                    continue
                
                if not instruction:
                    continue
                
                # Optional input
                input_text = input("📝 Input (optional): ").strip()
                
                print("\n🤖 Generating code...")
                
                # Generate code
                responses = self.generate_code(
                    instruction=instruction,
                    input_text=input_text,
                    max_new_tokens=512,
                    temperature=0.7,
                )
                
                print("\n" + "="*50)
                print("🎯 Generated Code:")
                print("="*50)
                print(responses[0])
                print("="*50)

                # Ask if user wants to execute the code
                if self.sandbox:
                    execute_choice = input("\n▶️ Execute this code? (y/n/s for subprocess): ").strip().lower()

                    if execute_choice in ['y', 'yes', 's', 'subprocess']:
                        test_input = input("📝 Test input (optional): ").strip()
                        use_subprocess = execute_choice in ['s', 'subprocess']

                        print(f"\n🔄 Executing code {'in subprocess' if use_subprocess else 'in sandbox'}...")

                        try:
                            if use_subprocess:
                                result = self.sandbox.execute_code_subprocess(responses[0], test_input)
                            else:
                                result = self.sandbox.execute_code(responses[0], test_input)

                            print("\n" + "="*50)
                            print("📊 Execution Results:")
                            print("="*50)

                            if result.success:
                                print("✅ Status: Success")
                                if result.output:
                                    print(f"📤 Output:\n{result.output}")
                                else:
                                    print("📤 Output: (no output)")
                            else:
                                print("❌ Status: Failed")
                                if result.error:
                                    print(f"🚨 Error:\n{result.error}")

                            print(f"⏱️ Execution time: {result.execution_time:.3f}s")
                            print("="*50)

                        except Exception as e:
                            print(f"❌ Execution error: {e}")
                else:
                    print("\n💡 Tip: Install code_sandbox.py to test generated code!")

                # Ask if user wants to score the code
                if self.scorer:
                    score_choice = input("\n🏆 Score this code? (y/n): ").strip().lower()

                    if score_choice in ['y', 'yes']:
                        category = input("📂 Category (general/functions/classes/algorithms/web/data/advanced): ").strip() or "general"

                        print("\n🔄 Scoring code...")

                        try:
                            score = self.scorer.score_code(instruction, responses[0], category)

                            print("\n" + "="*50)
                            print("🏆 Code Score Analysis:")
                            print("="*50)
                            print(f"🎯 Total Score: {score.total_score:.1f}/100 (Grade: {score.grade})")
                            print(f"📊 Breakdown:")
                            print(f"  - Syntax: {score.syntax_score:.1f}/100")
                            print(f"  - Functionality: {score.functionality_score:.1f}/100")
                            print(f"  - Style: {score.style_score:.1f}/100")
                            print(f"  - Efficiency: {score.efficiency_score:.1f}/100")
                            print(f"  - Documentation: {score.documentation_score:.1f}/100")

                            if score.positive_features:
                                print(f"\n✅ Strengths:")
                                for feature in score.positive_features:
                                    print(f"  • {feature}")

                            if score.suggestions:
                                print(f"\n💡 Suggestions:")
                                for suggestion in score.suggestions:
                                    print(f"  • {suggestion}")

                            print("="*50)

                            # Save scores
                            self.scorer.save_scores()

                        except Exception as e:
                            print(f"❌ Scoring error: {e}")
                else:
                    print("\n💡 Tip: Install scoring_system.py to score your code!")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show example prompts"""
        examples = [
            "Write a function to calculate fibonacci numbers",
            "Create a class for a binary search tree",
            "Implement a decorator for timing function execution",
            "Write a script to parse CSV files",
            "Create a function to validate email addresses",
            "Implement a simple web scraper using requests",
            "Write a function to merge two sorted lists",
            "Create a class for handling database connections",
        ]
        
        print("\n📚 Example Instructions:")
        print("-" * 30)
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("-" * 30)
        print("\n🏆 Special Commands:")
        print("• 'scores' or 'highscores' - View high scores")
        print("• 'stats' or 'statistics' - View performance statistics")

    def _show_high_scores(self):
        """Show high scores"""
        if not self.scorer:
            print("❌ Scoring system not available. Install scoring_system.py")
            return

        try:
            high_scores = self.scorer.get_high_scores(limit=10)

            if not high_scores:
                print("📊 No scores recorded yet. Generate and score some code first!")
                return

            print("\n🏆 HIGH SCORES LEADERBOARD")
            print("=" * 60)

            for i, score in enumerate(high_scores, 1):
                print(f"#{i:2d}. {score.grade} ({score.total_score:.1f}/100) - {score.category.title()}")
                print(f"     📝 {score.instruction[:50]}{'...' if len(score.instruction) > 50 else ''}")
                print(f"     📅 {score.timestamp[:10]}")
                print()

            print("=" * 60)

        except Exception as e:
            print(f"❌ Error loading high scores: {e}")

    def _show_statistics(self):
        """Show performance statistics"""
        if not self.scorer:
            print("❌ Scoring system not available. Install scoring_system.py")
            return

        try:
            stats = self.scorer.get_statistics()

            if "error" in stats:
                print(f"📊 {stats['error']}")
                return

            print("\n📈 PERFORMANCE STATISTICS")
            print("=" * 60)
            print(f"📊 Overall Performance:")
            print(f"   • Total Submissions: {stats['total_submissions']}")
            print(f"   • Average Score: {stats['average_score']:.1f}/100")
            print(f"   • Highest Score: {stats['highest_score']:.1f}/100")
            print(f"   • Recent Trend: {stats['recent_trend'].replace('_', ' ').title()}")

            print(f"\n🎓 Grade Distribution:")
            for grade, count in sorted(stats['grade_distribution'].items()):
                print(f"   • {grade}: {count}")

            if stats['category_performance']:
                print(f"\n📂 Category Performance:")
                for category, perf in stats['category_performance'].items():
                    print(f"   • {category.title()}: {perf['average']:.1f} avg ({perf['count']} submissions)")

            print("=" * 60)

        except Exception as e:
            print(f"❌ Error loading statistics: {e}")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Qwen3 Python Coder Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./qwen3-python-coder",
        help="Path to the trained LoRA model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model identifier"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Single instruction to process"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input context for the instruction"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    coder = Qwen3PythonCoder(
        model_path=args.model_path,
        base_model=args.base_model,
    )
    
    if args.interactive:
        # Start interactive mode
        coder.interactive_chat()
    elif args.instruction:
        # Process single instruction
        responses = coder.generate_code(
            instruction=args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        print("Generated Code:")
        print("=" * 50)
        print(responses[0])
        print("=" * 50)
    else:
        print("Please specify --interactive or --instruction")

if __name__ == "__main__":
    main()
