#!/usr/bin/env python3
"""
High Score System for Qwen3-8B Python Coder
Comprehensive scoring and ranking system for generated code quality
"""

import json
import time
import ast
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeScore:
    """Individual code score with detailed metrics"""
    instruction: str
    generated_code: str
    timestamp: str
    
    # Core metrics (0-100 each)
    syntax_score: float = 0.0
    functionality_score: float = 0.0
    style_score: float = 0.0
    efficiency_score: float = 0.0
    documentation_score: float = 0.0
    
    # Composite scores
    total_score: float = 0.0
    grade: str = "F"
    
    # Detailed analysis
    syntax_errors: List[str] = None
    style_issues: List[str] = None
    positive_features: List[str] = None
    suggestions: List[str] = None
    
    # Metadata
    code_length: int = 0
    complexity_estimate: str = "low"
    category: str = "general"
    
    def __post_init__(self):
        if self.syntax_errors is None:
            self.syntax_errors = []
        if self.style_issues is None:
            self.style_issues = []
        if self.positive_features is None:
            self.positive_features = []
        if self.suggestions is None:
            self.suggestions = []

class CodeScorer:
    """Advanced code scoring system"""
    
    def __init__(self):
        """Initialize the scorer"""
        self.scores_history = []
        self.high_scores = []
        self.categories = {
            "functions": "Basic Functions",
            "classes": "Object-Oriented Programming", 
            "algorithms": "Algorithms & Data Structures",
            "web": "Web Development",
            "data": "Data Processing",
            "advanced": "Advanced Python Features",
            "general": "General Programming"
        }
    
    def score_code(self, instruction: str, code: str, category: str = "general") -> CodeScore:
        """
        Comprehensive code scoring
        
        Args:
            instruction: The original instruction
            code: Generated Python code
            category: Code category for context
            
        Returns:
            CodeScore with detailed metrics
        """
        score = CodeScore(
            instruction=instruction,
            generated_code=code,
            timestamp=datetime.now().isoformat(),
            category=category,
            code_length=len(code)
        )
        
        # Calculate individual scores
        score.syntax_score = self._score_syntax(code, score)
        score.functionality_score = self._score_functionality(code, instruction, score)
        score.style_score = self._score_style(code, score)
        score.efficiency_score = self._score_efficiency(code, score)
        score.documentation_score = self._score_documentation(code, score)
        
        # Calculate composite score (weighted average)
        weights = {
            'syntax': 0.25,
            'functionality': 0.30,
            'style': 0.20,
            'efficiency': 0.15,
            'documentation': 0.10
        }
        
        score.total_score = (
            score.syntax_score * weights['syntax'] +
            score.functionality_score * weights['functionality'] +
            score.style_score * weights['style'] +
            score.efficiency_score * weights['efficiency'] +
            score.documentation_score * weights['documentation']
        )
        
        # Assign grade
        score.grade = self._assign_grade(score.total_score)
        
        # Estimate complexity
        score.complexity_estimate = self._estimate_complexity(code)
        
        # Add to history
        self.scores_history.append(score)
        
        # Update high scores
        self._update_high_scores(score)
        
        return score
    
    def _score_syntax(self, code: str, score: CodeScore) -> float:
        """Score syntax validity and correctness"""
        if not code.strip():
            score.syntax_errors.append("Empty code")
            return 0.0
        
        try:
            # Parse the AST
            tree = ast.parse(code)
            base_score = 80.0  # Base score for valid syntax
            
            # Check for common syntax patterns
            has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            has_classes = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
            
            if has_functions:
                base_score += 10
                score.positive_features.append("Contains function definitions")
            if has_classes:
                base_score += 10
                score.positive_features.append("Contains class definitions")
            if has_imports:
                base_score += 5
                score.positive_features.append("Uses imports appropriately")
            
            return min(base_score, 100.0)
            
        except SyntaxError as e:
            score.syntax_errors.append(f"Syntax error: {str(e)}")
            return 20.0  # Partial credit for attempting code
        except Exception as e:
            score.syntax_errors.append(f"Parse error: {str(e)}")
            return 10.0
    
    def _score_functionality(self, code: str, instruction: str, score: CodeScore) -> float:
        """Score how well the code addresses the instruction"""
        if not code.strip():
            return 0.0
        
        base_score = 50.0
        
        # Extract key terms from instruction
        instruction_lower = instruction.lower()
        code_lower = code.lower()
        
        # Common programming concepts
        concepts = {
            'function': ['def ', 'function'],
            'class': ['class ', 'object'],
            'loop': ['for ', 'while ', 'loop'],
            'condition': ['if ', 'elif ', 'else:', 'condition'],
            'list': ['list', 'array', '[]'],
            'dict': ['dict', 'dictionary', '{}'],
            'file': ['file', 'open(', 'read', 'write'],
            'sort': ['sort', 'sorted'],
            'search': ['search', 'find', 'index'],
            'calculate': ['calculate', 'compute', '+', '-', '*', '/'],
            'return': ['return'],
            'print': ['print(']
        }
        
        # Check if code implements requested concepts
        matches = 0
        total_concepts = 0
        
        for concept, keywords in concepts.items():
            if concept in instruction_lower:
                total_concepts += 1
                if any(keyword in code_lower for keyword in keywords):
                    matches += 1
                    base_score += 10
                    score.positive_features.append(f"Implements {concept}")
                else:
                    score.suggestions.append(f"Consider implementing {concept}")
        
        # Bonus for comprehensive implementation
        if total_concepts > 0 and matches == total_concepts:
            base_score += 20
            score.positive_features.append("Addresses all key requirements")
        
        return min(base_score, 100.0)
    
    def _score_style(self, code: str, score: CodeScore) -> float:
        """Score code style and readability"""
        if not code.strip():
            return 0.0
        
        base_score = 60.0
        lines = code.split('\n')
        
        # Check indentation consistency
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.append(indent)
        
        if indents and len(set(indents)) <= 2:  # Consistent indentation
            base_score += 10
            score.positive_features.append("Consistent indentation")
        elif indents:
            score.style_issues.append("Inconsistent indentation")
        
        # Check for meaningful variable names
        try:
            tree = ast.parse(code)
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.append(node.id)
            
            meaningful_names = [name for name in names if len(name) > 2 and not name.startswith('_')]
            if len(meaningful_names) > len(names) * 0.7:
                base_score += 10
                score.positive_features.append("Meaningful variable names")
            else:
                score.style_issues.append("Consider more descriptive variable names")
        except:
            pass
        
        # Check for comments
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if comment_lines:
            base_score += 10
            score.positive_features.append("Includes comments")
        
        # Check line length (PEP 8)
        long_lines = [line for line in lines if len(line) > 79]
        if not long_lines:
            base_score += 5
            score.positive_features.append("Follows line length guidelines")
        elif len(long_lines) > len(lines) * 0.3:
            score.style_issues.append("Some lines are too long (>79 characters)")
        
        return min(base_score, 100.0)
    
    def _score_efficiency(self, code: str, score: CodeScore) -> float:
        """Score code efficiency and best practices"""
        if not code.strip():
            return 0.0
        
        base_score = 70.0
        code_lower = code.lower()
        
        # Check for efficient patterns
        efficient_patterns = {
            'list_comprehension': r'\[.*for.*in.*\]',
            'generator': r'\(.*for.*in.*\)',
            'enumerate': r'enumerate\(',
            'zip': r'zip\(',
            'join': r'\.join\(',
            'set_operations': r'set\(',
        }
        
        for pattern_name, pattern in efficient_patterns.items():
            if re.search(pattern, code):
                base_score += 5
                score.positive_features.append(f"Uses {pattern_name.replace('_', ' ')}")
        
        # Check for inefficient patterns
        inefficient_patterns = {
            'string_concatenation_loop': (r'for.*\+.*str', "Consider using join() for string concatenation in loops"),
            'nested_loops': (r'for.*for.*', "Consider optimizing nested loops if possible"),
        }
        
        for pattern_name, (pattern, suggestion) in inefficient_patterns.items():
            if re.search(pattern, code):
                base_score -= 10
                score.suggestions.append(suggestion)
        
        return max(min(base_score, 100.0), 0.0)
    
    def _score_documentation(self, code: str, score: CodeScore) -> float:
        """Score documentation quality"""
        if not code.strip():
            return 0.0
        
        base_score = 40.0
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            base_score += 30
            score.positive_features.append("Includes docstrings")
        
        # Check for comments
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        if comment_lines > 0:
            base_score += 20
            score.positive_features.append("Includes inline comments")
        
        # Check for type hints
        if ':' in code and '->' in code:
            base_score += 10
            score.positive_features.append("Uses type hints")
        
        return min(base_score, 100.0)
    
    def _assign_grade(self, total_score: float) -> str:
        """Assign letter grade based on total score"""
        if total_score >= 90:
            return "A+"
        elif total_score >= 85:
            return "A"
        elif total_score >= 80:
            return "A-"
        elif total_score >= 75:
            return "B+"
        elif total_score >= 70:
            return "B"
        elif total_score >= 65:
            return "B-"
        elif total_score >= 60:
            return "C+"
        elif total_score >= 55:
            return "C"
        elif total_score >= 50:
            return "C-"
        elif total_score >= 45:
            return "D+"
        elif total_score >= 40:
            return "D"
        else:
            return "F"
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity"""
        lines = len([line for line in code.split('\n') if line.strip()])
        
        if lines < 10:
            return "low"
        elif lines < 30:
            return "medium"
        else:
            return "high"
    
    def _update_high_scores(self, score: CodeScore):
        """Update high scores list"""
        self.high_scores.append(score)
        
        # Sort by total score (descending) and keep top 10
        self.high_scores.sort(key=lambda x: x.total_score, reverse=True)
        self.high_scores = self.high_scores[:10]
    
    def get_high_scores(self, category: str = None, limit: int = 10) -> List[CodeScore]:
        """Get high scores, optionally filtered by category"""
        scores = self.high_scores
        
        if category:
            scores = [s for s in scores if s.category == category]
        
        return scores[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if not self.scores_history:
            return {"error": "No scores recorded yet"}
        
        scores = [s.total_score for s in self.scores_history]
        
        stats = {
            "total_submissions": len(self.scores_history),
            "average_score": sum(scores) / len(scores),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "grade_distribution": {},
            "category_performance": {},
            "recent_trend": self._calculate_trend(),
        }
        
        # Grade distribution
        grades = [s.grade for s in self.scores_history]
        for grade in set(grades):
            stats["grade_distribution"][grade] = grades.count(grade)
        
        # Category performance
        for category in self.categories:
            category_scores = [s.total_score for s in self.scores_history if s.category == category]
            if category_scores:
                stats["category_performance"][category] = {
                    "count": len(category_scores),
                    "average": sum(category_scores) / len(category_scores),
                    "best": max(category_scores)
                }
        
        return stats
    
    def _calculate_trend(self) -> str:
        """Calculate recent performance trend"""
        if len(self.scores_history) < 5:
            return "insufficient_data"
        
        recent_scores = [s.total_score for s in self.scores_history[-5:]]
        older_scores = [s.total_score for s in self.scores_history[-10:-5]] if len(self.scores_history) >= 10 else []
        
        if not older_scores:
            return "insufficient_data"
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg + 5:
            return "improving"
        elif recent_avg < older_avg - 5:
            return "declining"
        else:
            return "stable"
    
    def save_scores(self, filename: str = "code_scores.json"):
        """Save scores to file"""
        data = {
            "scores_history": [asdict(score) for score in self.scores_history],
            "high_scores": [asdict(score) for score in self.high_scores],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Scores saved to {filename}")
    
    def load_scores(self, filename: str = "code_scores.json"):
        """Load scores from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.scores_history = [CodeScore(**score_data) for score_data in data.get("scores_history", [])]
            self.high_scores = [CodeScore(**score_data) for score_data in data.get("high_scores", [])]
            
            logger.info(f"Scores loaded from {filename}")
        except FileNotFoundError:
            logger.info(f"No existing scores file found: {filename}")
        except Exception as e:
            logger.error(f"Error loading scores: {e}")

def main():
    """Demo the scoring system"""
    scorer = CodeScorer()
    
    # Example code samples
    test_cases = [
        {
            "instruction": "Write a function to calculate factorial",
            "code": '''def factorial(n):
    """Calculate factorial of a number."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test the function
print(factorial(5))''',
            "category": "functions"
        },
        {
            "instruction": "Create a simple calculator class",
            "code": '''class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b''',
            "category": "classes"
        }
    ]
    
    print("🏆 Code Scoring System Demo")
    print("=" * 50)
    
    for test in test_cases:
        score = scorer.score_code(
            test["instruction"], 
            test["code"], 
            test["category"]
        )
        
        print(f"\n📝 Instruction: {score.instruction}")
        print(f"🎯 Total Score: {score.total_score:.1f}/100 (Grade: {score.grade})")
        print(f"📊 Breakdown:")
        print(f"  - Syntax: {score.syntax_score:.1f}/100")
        print(f"  - Functionality: {score.functionality_score:.1f}/100")
        print(f"  - Style: {score.style_score:.1f}/100")
        print(f"  - Efficiency: {score.efficiency_score:.1f}/100")
        print(f"  - Documentation: {score.documentation_score:.1f}/100")
        
        if score.positive_features:
            print(f"✅ Strengths: {', '.join(score.positive_features)}")
        if score.suggestions:
            print(f"💡 Suggestions: {', '.join(score.suggestions)}")
        
        print("-" * 30)
    
    # Show statistics
    stats = scorer.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"Average Score: {stats['average_score']:.1f}")
    print(f"Best Score: {stats['highest_score']:.1f}")

if __name__ == "__main__":
    main()
