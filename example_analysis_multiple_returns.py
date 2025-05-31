"""
Analysis with Multiple Returns Example
=====================================
This example shows how to use funkydspy with functions that return multiple values.
The function analyzes a list of numbers and returns both statistical information and filtered data.
"""
#%%
import funkydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.funky
def analyse(numbers: list[float], threshold: float):
    """
    Analyze a list of numbers and return statistics and filtered values.
    
    Inputs
    ------
    numbers    : list[float]   (values to summarise)
    threshold  : float         (split point)

    Returns
    -------
    mean_value       : float           average of *numbers*
    above_threshold  : list[float]     values > threshold
    """
    # body never executed
    pass
#%%
data = [3.5, 7.2, 1.8, 9.0]
cut_off = 4.0

# Use fd.predict for direct prediction without chain-of-thought
(data, cut_off) | analyse | fd.predict

# %%
