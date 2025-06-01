from typing import List
import funnydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', cache=False))

print("=== Testing Tuple Variable Name Extraction ===")

@fd.funky
def analyse_tuple(
    nums: List[float],  # list of numbers
    threshold: float   # split point
) -> tuple[float, List[float]]:
    """
    Analyze numbers and return structured statistics.
    """
    mean = "The average"
    above = "Numbers above threshold"
    return mean, above                       # body never runs

print("Signature:", analyse_tuple.signature)
print("Output fields:", analyse_tuple.signature.output_fields)
print()

print("Testing with _prediction=True:")
result = analyse_tuple([3, 7, 1, 9], 4, _prediction=True)
print('Result:', result)
print('Expected field names: mean, above (not field0, field1)')
print('✅ WORKING!' if 'mean' in str(result) and 'above' in str(result) else '❌ NOT WORKING')
print()

print("Testing without _prediction:")
result2 = analyse_tuple([3, 7, 1, 9], 4)
print('Result2:', result2)
print('Result2 type:', type(result2)) 