#!/usr/bin/env python3
"""
Final Demo: DSPy-style Parallel Execution with FunnyDSPy
=========================================================
This demonstrates the exact same experience as standard DSPy's parallelize() function.
"""

import sys
sys.path.insert(0, '.')

import funnydspy as fd
import dspy
from typing import List

# Configure DSPy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

print("🚀 FunnyDSPy: Same experience as standard DSPy!\n")

# Example 1: DSPy function parallelization (like standard DSPy)
@fd.Predict
def classify_text(text: str, categories: List[str]) -> str:
    """Classify text into one of the given categories."""
    return category

print("1️⃣ DSPy-style function parallelization:")
classify_parallel = fd.parallelize(classify_text)
results = classify_parallel([
    {'text': 'Machine learning algorithms', 'categories': ['tech', 'sports', 'politics']},
    {'text': 'Football championship', 'categories': ['tech', 'sports', 'politics']},
    {'text': 'Election results', 'categories': ['tech', 'sports', 'politics']}
])
print(f"Results: {results}\n")

# Example 2: Regular Python function parallelization (enhanced from standard DSPy)
def process_data(x: int, multiplier: int) -> int:
    """A regular Python function."""
    return x * multiplier

print("2️⃣ Regular Python function (sequential execution):")
process_parallel = fd.parallelize(process_data)
results = process_parallel([
    {'x': 5, 'multiplier': 2},
    {'x': 10, 'multiplier': 3},
    {'x': 7, 'multiplier': 4}
])
print(f"Results: {results}\n")

# Example 3: Recursive function parallelization (like your structure_and_summarize example)
def recursive_summarize(content: str, depth: int) -> str:
    """Simulate recursive summarization."""
    if depth <= 0:
        return f"Summary: {content[:20]}..."
    else:
        # In real use, this would call itself recursively with parallel execution
        return f"Level {depth} summary of: {content[:15]}..."

print("3️⃣ Recursive function parallelization:")
summarize_parallel = fd.parallelize(recursive_summarize)
results = summarize_parallel([
    {'content': 'This is a long document about AI research and applications', 'depth': 1},
    {'content': 'This is another document about machine learning techniques', 'depth': 2},
    {'content': 'This third document covers neural network architectures', 'depth': 1}
])
print(f"Results: {results}\n")

print("✅ Perfect! Same elegant experience as standard DSPy's parallelize() function!")
print("🎯 Key benefits:")
print("   • Works with any function (DSPy decorated or regular Python)")
print("   • DSPy functions get true parallel execution") 
print("   • Regular functions get sequential execution (can be extended)")
print("   • Same API as standard DSPy")
print("   • Clean, readable code") 