#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

import funnydspy as fd
import dspy

# Configure DSPy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

# Test function
@fd.Predict  
def test_func(chunk: str) -> str: 
    return gist

print("Testing single call...")
single_result = test_func(chunk='test content')
print(f"Single result: {single_result}")
print(f"Type: {type(single_result)}")

print("\nTesting parallel call...")
try:
    results = fd.parallel(test_func, [{'chunk': 'test1'}, {'chunk': 'test2'}])
    print(f"Parallel results: {results}")
    print(f"Type of first result: {type(results[0])}")
    print(f"First result value: {results[0]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 