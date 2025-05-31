"""
NamedTuple Analysis Example
===========================
This example demonstrates using NamedTuple for structured output from funkydspy functions.
Shows how to define typed return values and use different DSPy modules (cot vs predict).
"""
# %%
from typing import List
import fcspy as fd
import dspy

# %%
# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

#%%
@fd.funky
def analyse(numbers: List[float],  # (values to summarise)
            threshold: float      # (split point)
            ):
    """
    Analyze numbers and return structured statistics.
    """
    mean = "The average of the numbers"
    above = "Numbers above the threshold"
    return mean, above

# %%
# Direct call using default Predict
result = analyse([3, 7, 1, 9], 4)
print(result)

# %%
# Using Chain of Thought via pipeline
result_cot = ([3, 7, 1, 9], 4) | analyse | fd.cot
print(result_cot)

# %%
# Using regular Predict via pipeline  
result_predict = ([3, 7, 1, 9], 4) | analyse
print(result_predict)
