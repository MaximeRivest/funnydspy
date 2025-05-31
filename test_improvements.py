"""
Test the improvements: reasoning preservation and type conversion
"""
from typing import List
from dataclasses import dataclass
import fcspy as fd
import dspy

# Configure DSPy (you'll need to set up your LM)
# dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', cache=False))

@dataclass
class Stats:
    mean: float
    above: List[float]

@fd.funky
def analyse(numbers: List[float],  # list of numbers to analyze
            threshold: float       # cutoff value
            ) -> Stats:
    """Analyze numbers and return statistics."""
    return Stats

# Test direct call with type conversion
print("=== Direct Call Test ===")
result = analyse([3, 7, 1, 9], 4)
print(f"Result: {result}")
print(f"Mean type: {type(result.mean)}")
if hasattr(result, 'above') and result.above:
    print(f"Above type: {type(result.above)}")
    if result.above:
        print(f"Above[0] type: {type(result.above[0])}")

print("\n=== Pipeline with Chain of Thought Test ===")
# Test pipeline with reasoning preservation
result_cot = ([3, 7, 1, 9], 4) | analyse | fd.cot
print(f"Result: {result_cot}")
print(f"Has reasoning: {'reasoning' in dict(result_cot)}")
if 'reasoning' in dict(result_cot):
    print(f"Reasoning: {result_cot.reasoning}")
print(f"Mean type: {type(result_cot.mean)}")
if hasattr(result_cot, 'above') and result_cot.above:
    print(f"Above type: {type(result_cot.above)}")
    if result_cot.above:
        print(f"Above[0] type: {type(result_cot.above[0])}") 