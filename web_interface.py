#!/usr/bin/env python3
"""
Gradio Web Interface for Qwen3-8B Python Coder
Provides an easy-to-use web UI for testing the fine-tuned model
"""

import gradio as gr
import torch
import logging
import time
import json
from typing import List, Tuple
import os

# Import our inference class, sandbox, and scoring system
from inference_script import Qwen3PythonCoder
from code_sandbox import CodeSandbox, ExecutionResult
from scoring_system import CodeScorer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebInterface:
    """Web interface wrapper for the Qwen3 Python Coder"""
    
    def __init__(self, model_path: str = "./qwen3-python-coder"):
        """Initialize the web interface"""
        self.model_path = model_path
        self.coder = None
        self.sandbox = CodeSandbox(timeout=10, max_memory_mb=100)
        self.scorer = CodeScorer()
        self.is_loaded = False

        # Load existing scores
        self.scorer.load_scores()
        
    def load_model(self) -> str:
        """Load the model and return status message"""
        try:
            logger.info("Loading Qwen3 Python Coder model...")
            self.coder = Qwen3PythonCoder(model_path=self.model_path)
            self.is_loaded = True
            return "✅ Model loaded successfully! Ready to generate Python code."
        except Exception as e:
            error_msg = f"❌ Failed to load model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def generate_code(
        self,
        instruction: str,
        input_text: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_outputs: int = 1,
    ) -> Tuple[str, str]:
        """
        Generate code and return result with metadata
        
        Returns:
            Tuple of (generated_code, metadata)
        """
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first.", ""
        
        if not instruction.strip():
            return "❌ Please provide an instruction.", ""
        
        try:
            start_time = time.time()
            
            # Generate code
            responses = self.coder.generate_code(
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_outputs,
            )
            
            generation_time = time.time() - start_time
            
            # Format output
            if num_outputs == 1:
                generated_code = responses[0]
            else:
                generated_code = "\n\n" + "="*50 + "\n\n".join(
                    [f"Output {i+1}:\n{resp}" for i, resp in enumerate(responses)]
                )
            
            # Create metadata
            metadata = f"""
📊 Generation Info:
• Time: {generation_time:.2f}s
• Tokens: ~{max_tokens} max
• Temperature: {temperature}
• Top-p: {top_p}
• Outputs: {num_outputs}
• Model: Qwen3-8B + LoRA
            """.strip()
            
            return generated_code, metadata
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, ""

    def execute_generated_code(
        self,
        code: str,
        test_input: str = "",
        use_subprocess: bool = False
    ) -> Tuple[str, str]:
        """
        Execute generated code in sandbox and return results

        Returns:
            Tuple of (execution_output, execution_info)
        """
        if not code.strip():
            return "❌ No code to execute.", ""

        try:
            # Execute in sandbox
            if use_subprocess:
                result = self.sandbox.execute_code_subprocess(code, test_input)
            else:
                result = self.sandbox.execute_code(code, test_input)

            # Format output
            if result.success:
                output = f"✅ Execution Successful!\n\n"
                if result.output:
                    output += f"Output:\n{result.output}\n"
                else:
                    output += "No output produced.\n"
            else:
                output = f"❌ Execution Failed!\n\n"
                if result.error:
                    output += f"Error:\n{result.error}\n"

            # Format execution info
            info = f"""
🔍 Execution Details:
• Success: {result.success}
• Time: {result.execution_time:.3f}s
• Method: {'Subprocess' if use_subprocess else 'In-process'}
• Timeout: {self.sandbox.timeout}s
            """.strip()

            return output, info

        except Exception as e:
            error_msg = f"❌ Sandbox error: {str(e)}"
            return error_msg, ""

    def score_generated_code(
        self,
        instruction: str,
        code: str,
        category: str = "general"
    ) -> Tuple[str, str]:
        """
        Score generated code and return detailed analysis

        Returns:
            Tuple of (score_summary, detailed_analysis)
        """
        if not code.strip():
            return "❌ No code to score.", ""

        try:
            # Score the code
            score = self.scorer.score_code(instruction, code, category)

            # Format score summary
            summary = f"""
🏆 **Code Score: {score.total_score:.1f}/100 (Grade: {score.grade})**

📊 **Breakdown:**
• Syntax: {score.syntax_score:.1f}/100
• Functionality: {score.functionality_score:.1f}/100
• Style: {score.style_score:.1f}/100
• Efficiency: {score.efficiency_score:.1f}/100
• Documentation: {score.documentation_score:.1f}/100

🎯 **Complexity:** {score.complexity_estimate.title()}
📂 **Category:** {score.category.title()}
            """.strip()

            # Format detailed analysis
            analysis = ""

            if score.positive_features:
                analysis += "✅ **Strengths:**\n"
                for feature in score.positive_features:
                    analysis += f"• {feature}\n"
                analysis += "\n"

            if score.syntax_errors:
                analysis += "❌ **Syntax Issues:**\n"
                for error in score.syntax_errors:
                    analysis += f"• {error}\n"
                analysis += "\n"

            if score.style_issues:
                analysis += "⚠️ **Style Issues:**\n"
                for issue in score.style_issues:
                    analysis += f"• {issue}\n"
                analysis += "\n"

            if score.suggestions:
                analysis += "💡 **Suggestions:**\n"
                for suggestion in score.suggestions:
                    analysis += f"• {suggestion}\n"
                analysis += "\n"

            # Save scores
            self.scorer.save_scores()

            return summary, analysis

        except Exception as e:
            error_msg = f"❌ Scoring error: {str(e)}"
            return error_msg, ""

    def get_high_scores_display(self) -> str:
        """Get formatted high scores display"""
        try:
            high_scores = self.scorer.get_high_scores(limit=5)

            if not high_scores:
                return "No scores recorded yet. Generate and score some code to see high scores!"

            display = "🏆 **Top 5 High Scores:**\n\n"

            for i, score in enumerate(high_scores, 1):
                display += f"**#{i}. {score.grade} ({score.total_score:.1f}/100)**\n"
                display += f"📝 {score.instruction[:60]}{'...' if len(score.instruction) > 60 else ''}\n"
                display += f"📅 {score.timestamp[:10]}\n\n"

            return display

        except Exception as e:
            return f"❌ Error loading high scores: {str(e)}"

    def get_statistics_display(self) -> str:
        """Get formatted statistics display"""
        try:
            stats = self.scorer.get_statistics()

            if "error" in stats:
                return stats["error"]

            display = f"""
📈 **Performance Statistics:**

📊 **Overall:**
• Total Submissions: {stats['total_submissions']}
• Average Score: {stats['average_score']:.1f}/100
• Highest Score: {stats['highest_score']:.1f}/100
• Recent Trend: {stats['recent_trend'].replace('_', ' ').title()}

🎓 **Grade Distribution:**
"""

            for grade, count in sorted(stats['grade_distribution'].items()):
                display += f"• {grade}: {count}\n"

            if stats['category_performance']:
                display += "\n📂 **Category Performance:**\n"
                for category, perf in stats['category_performance'].items():
                    display += f"• {category.title()}: {perf['average']:.1f} avg ({perf['count']} submissions)\n"

            return display.strip()

        except Exception as e:
            return f"❌ Error loading statistics: {str(e)}"

# Global interface instance
web_interface = WebInterface()

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .code-output {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 10px;
    }
    """
    
    with gr.Blocks(css=css, title="Qwen3 Python Coder", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🐍 Qwen3-8B Python Coder
        ### Fine-tuned AI Assistant for Python Code Generation
        
        This model has been fine-tuned on 18k Python code instructions to help you generate high-quality Python code.
        """)
        
        # Model loading section
        with gr.Row():
            with gr.Column(scale=3):
                load_btn = gr.Button("🚀 Load Model", variant="primary", size="lg")
            with gr.Column(scale=7):
                load_status = gr.Textbox(
                    label="Model Status",
                    value="🔄 Click 'Load Model' to initialize the AI coder",
                    interactive=False
                )
        
        gr.Markdown("---")
        
        # Main interface
        with gr.Tabs():
            # Generation Tab
            with gr.TabItem("🎯 Generate Code"):
                with gr.Row():
                    # Input column
                    with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")
                
                instruction = gr.Textbox(
                    label="Instruction",
                    placeholder="e.g., Write a function to calculate fibonacci numbers",
                    lines=3,
                    max_lines=5
                )
                
                input_text = gr.Textbox(
                    label="Input (Optional)",
                    placeholder="e.g., [1, 2, 3, 4, 5] or additional context",
                    lines=2,
                    max_lines=3
                )
                
                # Generation parameters
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=1024,
                        value=512,
                        step=50,
                        label="Max Tokens"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (creativity)"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (nucleus sampling)"
                    )
                    
                    num_outputs = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=1,
                        step=1,
                        label="Number of outputs"
                    )
                
                        code_category = gr.Dropdown(
                            choices=["general", "functions", "classes", "algorithms", "web", "data", "advanced"],
                            value="general",
                            label="Code Category"
                        )

                        generate_btn = gr.Button("🎯 Generate Code", variant="primary", size="lg")

                # Output column
                with gr.Column(scale=2):
                gr.Markdown("### 🤖 Generated Code")
                
                generated_code = gr.Textbox(
                    label="Python Code",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True,
                    elem_classes=["code-output"]
                )

                # Code execution section
                with gr.Row():
                    with gr.Column(scale=1):
                        test_input = gr.Textbox(
                            label="Test Input (optional)",
                            placeholder="Input for your code (e.g., test data)",
                            lines=2,
                            max_lines=3
                        )

                        with gr.Row():
                            execute_btn = gr.Button("▶️ Run Code", variant="secondary")
                            score_btn = gr.Button("🏆 Score Code", variant="secondary")
                            subprocess_check = gr.Checkbox(
                                label="Use subprocess (safer)",
                                value=True
                            )

                    with gr.Column(scale=2):
                        execution_output = gr.Textbox(
                            label="Execution Output",
                            lines=6,
                            max_lines=10,
                            elem_classes=["code-output"]
                        )

                        score_summary = gr.Textbox(
                            label="Code Score Summary",
                            lines=6,
                            max_lines=10,
                            elem_classes=["code-output"]
                        )

                with gr.Row():
                    metadata = gr.Textbox(
                        label="Generation Info",
                        lines=4,
                        interactive=False,
                        scale=1
                    )

                    execution_info = gr.Textbox(
                        label="Execution Info",
                        lines=3,
                        interactive=False,
                        scale=1
                    )

                    score_details = gr.Textbox(
                        label="Score Analysis",
                        lines=3,
                        interactive=False,
                        scale=1
                    )
        
                # Example prompts
                gr.Markdown("### 💡 Example Prompts")
        
        examples = [
            ["Write a function to calculate the factorial of a number", ""],
            ["Create a class for a simple calculator", ""],
            ["Implement a binary search algorithm", "[1, 3, 5, 7, 9, 11, 13, 15]"],
            ["Write a decorator to measure function execution time", ""],
            ["Create a function to validate email addresses using regex", ""],
            ["Implement a simple web scraper using requests and BeautifulSoup", "https://example.com"],
            ["Write a function to merge two sorted lists", "list1 = [1, 3, 5], list2 = [2, 4, 6]"],
            ["Create a context manager for file operations", ""],
        ]
        
                gr.Examples(
                    examples=examples,
                    inputs=[instruction, input_text],
                    label="Click any example to try it:"
                )

            # High Scores Tab
            with gr.TabItem("🏆 High Scores"):
                with gr.Row():
                    refresh_scores_btn = gr.Button("🔄 Refresh", variant="secondary")

                high_scores_display = gr.Markdown(
                    value="Loading high scores...",
                    label="High Scores"
                )

            # Statistics Tab
            with gr.TabItem("📊 Statistics"):
                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 Refresh", variant="secondary")

                statistics_display = gr.Markdown(
                    value="Loading statistics...",
                    label="Performance Statistics"
                )
        
        # Event handlers
        load_btn.click(
            fn=web_interface.load_model,
            outputs=load_status
        )
        
        generate_btn.click(
            fn=web_interface.generate_code,
            inputs=[instruction, input_text, max_tokens, temperature, top_p, num_outputs],
            outputs=[generated_code, metadata]
        )

        execute_btn.click(
            fn=web_interface.execute_generated_code,
            inputs=[generated_code, test_input, subprocess_check],
            outputs=[execution_output, execution_info]
        )

        score_btn.click(
            fn=web_interface.score_generated_code,
            inputs=[instruction, generated_code, code_category],
            outputs=[score_summary, score_details]
        )

        refresh_scores_btn.click(
            fn=web_interface.get_high_scores_display,
            outputs=high_scores_display
        )

        refresh_stats_btn.click(
            fn=web_interface.get_statistics_display,
            outputs=statistics_display
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📚 Tips for Better Results:
        - Be specific in your instructions
        - Provide examples or context when helpful
        - Adjust temperature: lower (0.3-0.5) for more focused code, higher (0.7-1.0) for more creative solutions
        - Use the input field for additional context, examples, or constraints
        - **Test your code**: Use the "Run Code" button to verify generated code works correctly
        - **Sandbox Safety**: Code execution is sandboxed and restricted for security

        ### 🔒 Sandbox Security:
        - Blocks dangerous operations (file access, network, subprocess)
        - 10-second execution timeout
        - Memory usage limits
        - Safe built-in functions only

        ### 🏆 Scoring System:
        - Comprehensive code quality analysis
        - Syntax, functionality, style, efficiency, and documentation scores
        - Letter grades (A+ to F) with detailed feedback
        - High scores leaderboard and performance statistics
        - Track your improvement over time

        **Model:** Qwen3-8B fine-tuned with LoRA on Python code instructions
        """)
    
    return interface

def main():
    """Launch the web interface"""
    interface = create_interface()
    
    # Launch with configuration
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
        show_tips=True,
        enable_queue=True,
        max_threads=4,
    )

if __name__ == "__main__":
    main()
