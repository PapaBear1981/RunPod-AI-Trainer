#!/usr/bin/env python3
"""
Comprehensive Test Suite for Qwen3-8B Python Coder
Evaluates the model's Python coding capabilities across different categories
"""

import json
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import re
import ast

from inference_script import Qwen3PythonCoder
from code_sandbox import CodeSandbox

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Individual test case"""
    category: str
    instruction: str
    input_text: str = ""
    expected_keywords: List[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    description: str = ""

@dataclass
class TestResult:
    """Test result with evaluation metrics"""
    test_case: TestCase
    generated_code: str
    execution_time: float
    syntax_valid: bool
    contains_keywords: bool
    estimated_quality: str  # poor, fair, good, excellent
    execution_success: bool = False
    execution_output: str = ""
    execution_error: str = ""
    notes: str = ""

class PythonCodeEvaluator:
    """Evaluates generated Python code quality"""
    
    @staticmethod
    def check_syntax(code: str) -> bool:
        """Check if the generated code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    @staticmethod
    def check_keywords(code: str, keywords: List[str]) -> bool:
        """Check if code contains expected keywords"""
        if not keywords:
            return True
        
        code_lower = code.lower()
        return any(keyword.lower() in code_lower for keyword in keywords)
    
    @staticmethod
    def estimate_quality(code: str, test_case: TestCase) -> str:
        """Estimate code quality based on various factors"""
        if not code.strip():
            return "poor"
        
        # Check for basic structure
        has_function = "def " in code
        has_class = "class " in code
        has_docstring = '"""' in code or "'''" in code
        has_comments = "#" in code
        has_error_handling = "try:" in code or "except:" in code
        
        score = 0
        
        # Basic functionality
        if has_function or has_class:
            score += 2
        
        # Documentation
        if has_docstring:
            score += 2
        elif has_comments:
            score += 1
        
        # Error handling
        if has_error_handling:
            score += 1
        
        # Length and complexity
        lines = len([line for line in code.split('\n') if line.strip()])
        if lines >= 5:
            score += 1
        if lines >= 15:
            score += 1
        
        # Keywords match
        if PythonCodeEvaluator.check_keywords(code, test_case.expected_keywords):
            score += 2
        
        # Quality mapping
        if score >= 7:
            return "excellent"
        elif score >= 5:
            return "good"
        elif score >= 3:
            return "fair"
        else:
            return "poor"

class TestSuite:
    """Main test suite for evaluating the Python coder"""
    
    def __init__(self, model_path: str = "./qwen3-python-coder"):
        """Initialize test suite"""
        self.model_path = model_path
        self.coder = None
        self.sandbox = CodeSandbox(timeout=10)
        self.test_cases = self._create_test_cases()
        self.results = []
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases"""
        test_cases = [
            # Basic Functions
            TestCase(
                category="Basic Functions",
                instruction="Write a function to calculate the factorial of a number",
                expected_keywords=["def", "factorial", "return"],
                difficulty="easy",
                description="Basic recursive or iterative function"
            ),
            TestCase(
                category="Basic Functions",
                instruction="Create a function to check if a number is prime",
                expected_keywords=["def", "prime", "return", "for"],
                difficulty="easy",
                description="Number theory function with loop"
            ),
            TestCase(
                category="Basic Functions",
                instruction="Write a function to reverse a string",
                expected_keywords=["def", "reverse", "return"],
                difficulty="easy",
                description="String manipulation function"
            ),
            
            # Data Structures
            TestCase(
                category="Data Structures",
                instruction="Implement a stack class with push, pop, and peek methods",
                expected_keywords=["class", "Stack", "push", "pop", "peek"],
                difficulty="medium",
                description="Basic data structure implementation"
            ),
            TestCase(
                category="Data Structures",
                instruction="Create a binary search tree class with insert and search methods",
                expected_keywords=["class", "BinarySearchTree", "insert", "search"],
                difficulty="hard",
                description="Tree data structure with methods"
            ),
            TestCase(
                category="Data Structures",
                instruction="Implement a queue using two stacks",
                expected_keywords=["class", "Queue", "enqueue", "dequeue", "stack"],
                difficulty="medium",
                description="Queue implementation using stacks"
            ),
            
            # Algorithms
            TestCase(
                category="Algorithms",
                instruction="Implement binary search algorithm",
                input_text="[1, 3, 5, 7, 9, 11, 13, 15]",
                expected_keywords=["def", "binary_search", "while", "return"],
                difficulty="medium",
                description="Classic search algorithm"
            ),
            TestCase(
                category="Algorithms",
                instruction="Write a function to sort a list using quicksort",
                expected_keywords=["def", "quicksort", "partition", "return"],
                difficulty="hard",
                description="Divide and conquer sorting algorithm"
            ),
            TestCase(
                category="Algorithms",
                instruction="Implement the Fibonacci sequence using dynamic programming",
                expected_keywords=["def", "fibonacci", "dp", "return"],
                difficulty="medium",
                description="Dynamic programming approach"
            ),
            
            # Object-Oriented Programming
            TestCase(
                category="OOP",
                instruction="Create a Calculator class with basic arithmetic operations",
                expected_keywords=["class", "Calculator", "add", "subtract", "multiply", "divide"],
                difficulty="easy",
                description="Basic OOP with methods"
            ),
            TestCase(
                category="OOP",
                instruction="Design a BankAccount class with deposit, withdraw, and balance methods",
                expected_keywords=["class", "BankAccount", "deposit", "withdraw", "balance"],
                difficulty="medium",
                description="Real-world OOP example"
            ),
            TestCase(
                category="OOP",
                instruction="Create an Animal base class and Dog, Cat subclasses with inheritance",
                expected_keywords=["class", "Animal", "Dog", "Cat", "inherit"],
                difficulty="medium",
                description="Inheritance and polymorphism"
            ),
            
            # Advanced Features
            TestCase(
                category="Advanced",
                instruction="Write a decorator to measure function execution time",
                expected_keywords=["def", "decorator", "time", "wrapper"],
                difficulty="hard",
                description="Python decorators"
            ),
            TestCase(
                category="Advanced",
                instruction="Create a context manager for file operations",
                expected_keywords=["class", "__enter__", "__exit__", "with"],
                difficulty="hard",
                description="Context manager protocol"
            ),
            TestCase(
                category="Advanced",
                instruction="Implement a generator function for fibonacci numbers",
                expected_keywords=["def", "yield", "fibonacci", "generator"],
                difficulty="medium",
                description="Generator functions"
            ),
            
            # Practical Applications
            TestCase(
                category="Practical",
                instruction="Write a function to validate email addresses using regex",
                expected_keywords=["def", "email", "regex", "re", "return"],
                difficulty="medium",
                description="Regular expressions for validation"
            ),
            TestCase(
                category="Practical",
                instruction="Create a function to parse CSV data and return a list of dictionaries",
                expected_keywords=["def", "csv", "parse", "return"],
                difficulty="medium",
                description="Data processing function"
            ),
            TestCase(
                category="Practical",
                instruction="Write a simple web scraper using requests library",
                input_text="https://example.com",
                expected_keywords=["import", "requests", "def", "scrape"],
                difficulty="medium",
                description="Web scraping basics"
            ),

            # Error Handling
            TestCase(
                category="Error Handling",
                instruction="Write a function that handles division by zero gracefully",
                expected_keywords=["def", "try", "except", "ZeroDivisionError"],
                difficulty="easy",
                description="Exception handling basics"
            ),
        ]
        
        return test_cases
    
    def load_model(self) -> bool:
        """Load the model for testing"""
        try:
            logger.info("Loading model for testing...")
            self.coder = Qwen3PythonCoder(model_path=self.model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        logger.info(f"Running test: {test_case.instruction[:50]}...")
        
        start_time = time.time()
        
        try:
            # Generate code
            responses = self.coder.generate_code(
                instruction=test_case.instruction,
                input_text=test_case.input_text,
                max_new_tokens=512,
                temperature=0.7,
            )
            
            generated_code = responses[0]
            execution_time = time.time() - start_time
            
            # Evaluate the generated code
            syntax_valid = PythonCodeEvaluator.check_syntax(generated_code)
            contains_keywords = PythonCodeEvaluator.check_keywords(
                generated_code, test_case.expected_keywords
            )
            estimated_quality = PythonCodeEvaluator.estimate_quality(generated_code, test_case)

            # Test execution in sandbox
            execution_success = False
            execution_output = ""
            execution_error = ""

            if syntax_valid:
                try:
                    exec_result = self.sandbox.execute_code(generated_code, test_case.input_text)
                    execution_success = exec_result.success
                    execution_output = exec_result.output
                    execution_error = exec_result.error
                except Exception as e:
                    execution_error = f"Sandbox error: {str(e)}"

            result = TestResult(
                test_case=test_case,
                generated_code=generated_code,
                execution_time=execution_time,
                syntax_valid=syntax_valid,
                contains_keywords=contains_keywords,
                estimated_quality=estimated_quality,
                execution_success=execution_success,
                execution_output=execution_output,
                execution_error=execution_error,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return TestResult(
                test_case=test_case,
                generated_code="",
                execution_time=time.time() - start_time,
                syntax_valid=False,
                contains_keywords=False,
                estimated_quality="poor",
                notes=f"Error: {str(e)}"
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test cases"""
        if not self.coder:
            if not self.load_model():
                logger.error("Cannot run tests without model")
                return []
        
        logger.info(f"Running {len(self.test_cases)} test cases...")
        
        self.results = []
        for i, test_case in enumerate(self.test_cases, 1):
            logger.info(f"Test {i}/{len(self.test_cases)}: {test_case.category}")
            result = self.run_single_test(test_case)
            self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        # Calculate statistics
        total_tests = len(self.results)
        syntax_valid_count = sum(1 for r in self.results if r.syntax_valid)
        keyword_match_count = sum(1 for r in self.results if r.contains_keywords)
        execution_success_count = sum(1 for r in self.results if r.execution_success)
        
        quality_counts = {}
        category_stats = {}
        
        for result in self.results:
            # Quality distribution
            quality = result.estimated_quality
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Category statistics
            category = result.test_case.category
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "syntax_valid": 0,
                    "keyword_match": 0,
                    "execution_success": 0,
                    "avg_time": 0
                }
            
            category_stats[category]["total"] += 1
            if result.syntax_valid:
                category_stats[category]["syntax_valid"] += 1
            if result.contains_keywords:
                category_stats[category]["keyword_match"] += 1
            if result.execution_success:
                category_stats[category]["execution_success"] += 1
            category_stats[category]["avg_time"] += result.execution_time
        
        # Calculate averages
        for category in category_stats:
            if category_stats[category]["total"] > 0:
                category_stats[category]["avg_time"] /= category_stats[category]["total"]
        
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tests
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "syntax_valid": syntax_valid_count,
                "syntax_valid_rate": syntax_valid_count / total_tests * 100,
                "keyword_match": keyword_match_count,
                "keyword_match_rate": keyword_match_count / total_tests * 100,
                "execution_success": execution_success_count,
                "execution_success_rate": execution_success_count / total_tests * 100,
                "avg_execution_time": avg_execution_time,
            },
            "quality_distribution": quality_counts,
            "category_statistics": category_stats,
            "detailed_results": [
                {
                    "category": r.test_case.category,
                    "instruction": r.test_case.instruction,
                    "difficulty": r.test_case.difficulty,
                    "syntax_valid": r.syntax_valid,
                    "contains_keywords": r.contains_keywords,
                    "estimated_quality": r.estimated_quality,
                    "execution_time": r.execution_time,
                    "code_length": len(r.generated_code),
                    "notes": r.notes,
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_report(self, filename: str = "test_report.json"):
        """Save test report to file"""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to {filename}")
        return filename

def main():
    """Run the test suite"""
    print("🧪 Qwen3-8B Python Coder Test Suite")
    print("=" * 50)
    
    # Initialize test suite
    test_suite = TestSuite()
    
    # Run tests
    results = test_suite.run_all_tests()
    
    if not results:
        print("❌ No tests were run")
        return
    
    # Generate and display report
    report = test_suite.generate_report()
    
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Syntax Valid: {report['summary']['syntax_valid']} ({report['summary']['syntax_valid_rate']:.1f}%)")
    print(f"Keyword Match: {report['summary']['keyword_match']} ({report['summary']['keyword_match_rate']:.1f}%)")
    print(f"Avg Execution Time: {report['summary']['avg_execution_time']:.2f}s")
    
    print("\n🎯 Quality Distribution:")
    for quality, count in report['quality_distribution'].items():
        print(f"  {quality.capitalize()}: {count}")
    
    print("\n📂 Category Performance:")
    for category, stats in report['category_statistics'].items():
        syntax_rate = stats['syntax_valid'] / stats['total'] * 100
        keyword_rate = stats['keyword_match'] / stats['total'] * 100
        print(f"  {category}:")
        print(f"    Syntax Valid: {syntax_rate:.1f}%")
        print(f"    Keyword Match: {keyword_rate:.1f}%")
        print(f"    Avg Time: {stats['avg_time']:.2f}s")
    
    # Save detailed report
    report_file = test_suite.save_report()
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main()
