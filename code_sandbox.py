#!/usr/bin/env python3
"""
Secure Code Sandbox for Qwen3-8B Python Coder
Safely execute and verify generated Python code with security restrictions
"""

import ast
import sys
import io
import contextlib
import time
import signal
import subprocess
import tempfile
import os
import traceback
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: str
    error: str
    execution_time: float
    memory_usage: Optional[float] = None
    exit_code: Optional[int] = None

class SecurityError(Exception):
    """Raised when code contains potentially dangerous operations"""
    pass

class CodeSandbox:
    """Secure sandbox for executing generated Python code"""
    
    # Dangerous modules and functions to block
    BLOCKED_IMPORTS = {
        'os', 'subprocess', 'sys', 'shutil', 'glob', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'smtplib', 'telnetlib', 'webbrowser',
        'ctypes', 'multiprocessing', 'threading', 'asyncio', 'concurrent',
        'importlib', '__import__', 'eval', 'exec', 'compile', 'open',
        'file', 'input', 'raw_input'
    }
    
    BLOCKED_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file', 'input',
        'raw_input', 'reload', 'vars', 'locals', 'globals', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr'
    }
    
    BLOCKED_ATTRIBUTES = {
        '__class__', '__bases__', '__subclasses__', '__mro__', '__globals__',
        '__code__', '__func__', '__self__', '__module__', '__dict__'
    }
    
    def __init__(
        self,
        timeout: int = 10,
        max_memory_mb: int = 100,
        allow_imports: List[str] = None,
        enable_networking: bool = False
    ):
        """
        Initialize the sandbox
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            allow_imports: List of additional allowed imports
            enable_networking: Whether to allow network operations
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.enable_networking = enable_networking
        
        # Default safe imports
        self.allowed_imports = {
            'math', 'random', 'datetime', 'time', 'json', 'csv', 're',
            'collections', 'itertools', 'functools', 'operator', 'string',
            'decimal', 'fractions', 'statistics', 'hashlib', 'base64',
            'uuid', 'copy', 'pickle', 'struct', 'array', 'heapq', 'bisect'
        }
        
        if allow_imports:
            self.allowed_imports.update(allow_imports)
        
        if enable_networking:
            self.allowed_imports.update(['requests', 'urllib', 'http'])
    
    def _analyze_ast(self, code: str) -> None:
        """
        Analyze AST for potentially dangerous operations
        
        Args:
            code: Python code to analyze
            
        Raises:
            SecurityError: If dangerous operations are detected
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SecurityError(f"Syntax error: {e}")
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, sandbox):
                self.sandbox = sandbox
                self.imports = set()
                self.calls = set()
                self.attributes = set()
            
            def visit_Import(self, node):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    self.imports.add(module_name)
                    if (module_name in self.sandbox.BLOCKED_IMPORTS and 
                        module_name not in self.sandbox.allowed_imports):
                        raise SecurityError(f"Blocked import: {module_name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split('.')[0]
                    self.imports.add(module_name)
                    if (module_name in self.sandbox.BLOCKED_IMPORTS and 
                        module_name not in self.sandbox.allowed_imports):
                        raise SecurityError(f"Blocked import: {module_name}")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    self.calls.add(func_name)
                    if func_name in self.sandbox.BLOCKED_BUILTINS:
                        raise SecurityError(f"Blocked builtin function: {func_name}")
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                if isinstance(node.attr, str):
                    self.attributes.add(node.attr)
                    if node.attr in self.sandbox.BLOCKED_ATTRIBUTES:
                        raise SecurityError(f"Blocked attribute access: {node.attr}")
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace"""
        safe_builtins = {}
        
        # Allow safe built-in functions
        safe_builtin_names = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'hex',
            'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'map',
            'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 'range',
            'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'zip'
        }
        
        for name in safe_builtin_names:
            if hasattr(__builtins__, name):
                safe_builtins[name] = getattr(__builtins__, name)
        
        # Create safe globals
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }
        
        return safe_globals
    
    def execute_code(self, code: str, test_input: str = "") -> ExecutionResult:
        """
        Execute code in the sandbox
        
        Args:
            code: Python code to execute
            test_input: Optional input for the code
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        try:
            # Security analysis
            self._analyze_ast(code)
            
            # Prepare execution environment
            safe_globals = self._create_safe_globals()
            safe_locals = {}
            
            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            old_stdin = sys.stdin
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            stdin_capture = io.StringIO(test_input)
            
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                sys.stdin = stdin_capture
                
                # Set up timeout
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Code execution timed out after {self.timeout} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
                
                try:
                    # Execute the code
                    exec(code, safe_globals, safe_locals)
                    
                    execution_time = time.time() - start_time
                    output = stdout_capture.getvalue()
                    error = stderr_capture.getvalue()
                    
                    return ExecutionResult(
                        success=True,
                        output=output,
                        error=error,
                        execution_time=execution_time
                    )
                    
                except TimeoutError as e:
                    return ExecutionResult(
                        success=False,
                        output="",
                        error=str(e),
                        execution_time=self.timeout
                    )
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    
                    return ExecutionResult(
                        success=False,
                        output=stdout_capture.getvalue(),
                        error=error_msg,
                        execution_time=execution_time
                    )
                finally:
                    signal.alarm(0)  # Cancel the alarm
                    
            finally:
                # Restore standard streams
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                sys.stdin = old_stdin
                
        except SecurityError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security violation: {str(e)}",
                execution_time=0.0
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=f"Sandbox error: {str(e)}",
                execution_time=execution_time
            )
    
    def execute_code_subprocess(self, code: str, test_input: str = "") -> ExecutionResult:
        """
        Execute code in a separate subprocess for additional isolation
        
        Args:
            code: Python code to execute
            test_input: Optional input for the code
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        try:
            # Security analysis
            self._analyze_ast(code)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute in subprocess
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout
                )
                
                stdout, stderr = process.communicate(input=test_input, timeout=self.timeout)
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout,
                    error=stderr,
                    execution_time=execution_time,
                    exit_code=process.returncode
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Code execution timed out after {self.timeout} seconds",
                    execution_time=self.timeout
                )
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
                    
        except SecurityError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security violation: {str(e)}",
                execution_time=0.0
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=f"Subprocess execution error: {str(e)}",
                execution_time=execution_time
            )

def test_code_examples():
    """Test the sandbox with various code examples"""
    sandbox = CodeSandbox(timeout=5)
    
    test_cases = [
        # Safe code
        {
            "name": "Simple function",
            "code": """
def add(a, b):
    return a + b

result = add(5, 3)
print(f"5 + 3 = {result}")
""",
            "input": "",
            "should_succeed": True
        },
        
        # Safe code with input
        {
            "name": "Fibonacci function",
            "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
""",
            "input": "",
            "should_succeed": True
        },
        
        # Dangerous code - file access
        {
            "name": "File access (blocked)",
            "code": """
with open('/etc/passwd', 'r') as f:
    print(f.read())
""",
            "input": "",
            "should_succeed": False
        },
        
        # Dangerous code - subprocess
        {
            "name": "Subprocess (blocked)",
            "code": """
import subprocess
result = subprocess.run(['ls', '-la'], capture_output=True, text=True)
print(result.stdout)
""",
            "input": "",
            "should_succeed": False
        },
        
        # Safe imports
        {
            "name": "Safe imports",
            "code": """
import math
import random
import datetime

print(f"Pi: {math.pi}")
print(f"Random: {random.randint(1, 10)}")
print(f"Now: {datetime.datetime.now()}")
""",
            "input": "",
            "should_succeed": True
        }
    ]
    
    print("🧪 Testing Code Sandbox")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        result = sandbox.execute_code(test_case['code'], test_case['input'])
        
        success_match = result.success == test_case['should_succeed']
        status = "✅ PASS" if success_match else "❌ FAIL"
        
        print(f"{status} - Expected: {test_case['should_succeed']}, Got: {result.success}")
        print(f"Time: {result.execution_time:.3f}s")
        
        if result.output:
            print(f"Output: {result.output[:100]}...")
        if result.error:
            print(f"Error: {result.error[:100]}...")
        
        print("-" * 30)

def main():
    """Main function for testing"""
    test_code_examples()

if __name__ == "__main__":
    main()
