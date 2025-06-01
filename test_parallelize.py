#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

import funnydspy as fd
import dspy

# Configure DSPy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

# Test FunnyDSPy function
@fd.Predict  
def analyze_text(text: str) -> str: 
    return analysis

# Test regular Python function
def add_numbers(x: int, y: int) -> int:
    return x + y

print("=== Testing fd.parallelize() ===")

print("\n1. Testing with FunnyDSPy function...")
parallel_analyze = fd.parallelize(analyze_text)
try:
    results = parallel_analyze([{'text': 'test1'}, {'text': 'test2'}])
    print(f"Results: {results}")
    print(f"Types: {[type(r) for r in results]}")
except Exception as e:
    print(f"Error: {e}")

print("\n2. Testing with regular Python function...")
parallel_add = fd.parallelize(add_numbers)
try:
    results = parallel_add([{'x': 1, 'y': 2}, {'x': 3, 'y': 4}])
    print(f"Results: {results}")
    print(f"Types: {[type(r) for r in results]}")
except Exception as e:
    print(f"Error: {e}")

print("\n3. Testing recursive function (simulated)...")
def recursive_func(data: str, depth: int) -> str:
    if depth <= 0:
        return f"Base: {data}"
    else:
        # Simulate recursive call
        return f"Level {depth}: {data}"

parallel_recursive = fd.parallelize(recursive_func)
try:
    results = parallel_recursive([
        {'data': 'test1', 'depth': 1}, 
        {'data': 'test2', 'depth': 2}
    ])
    print(f"Results: {results}")
    print(f"Types: {[type(r) for r in results]}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== All tests completed ===") 