#!/usr/bin/env python3
"""
NamedTuple Analysis Example
===========================
This example demonstrates using NamedTuple for structured output from funnydspy functions.
Shows how to define typed return values and use different DSPy modules (cot vs predict).
"""
# %%
from typing import NamedTuple, List
from dataclasses import dataclass
import funnydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))


#%%
# support concise syntax
@fd.Predict
def rag(question,response): return response

rag("The number is 1", "What is the number?")



#%%
class analyse_og(dspy.Signature):
    """
    Analyze numbers and return structured statistics.
    """
    numbers: List[float] = dspy.InputField(description="values to summarise")
    threshold: float = dspy.InputField(description="split point")
    Stats_mean_value: float = dspy.OutputField(description="average of numbers")
    Stats_above_threshold: List[float] = dspy.OutputField(description="values > threshold")

dspy.Predict(analyse_og)(numbers=[3, 7, 1, 9], threshold=4)


#%%
# ==============================================
# Syntax 1: Using dataclase
# for multiple returns, and using python docstring
# when describing the inputs and outputs
# ==============================================
@dataclass
class Stats:
    """Structured output for analysis results."""
    mean_value: float
    above_threshold: List[float]


@fd.funky
def analyse1(numbers: List[float],
            threshold: float) -> Stats:
    """
    Analyze numbers and return structured statistics.

    Parameters
    ----------
    numbers: values to summarise
    threshold: split point

    Returns
    -------
    Stats.mean_value: average of numbers
    Stats.above_threshold: values > threshold
    """
    return Stats

analyse1([3, 7, 1, 9], 4)



#%%
#==============================================
# Syntax 1.1: Using dataclase but with docment
#==============================================
@dataclass
class Stats:
    """Structured output for analysis results."""
    mean_value: float # values to summarise
    above_threshold: List[float] # values > threshold


@fd.Predict
def analyse2(numbers: List[float], # values to summarise and I have a lot to say about this 
            threshold: float # split point
            ) -> Stats:
    """
    Analyze numbers and return structured statistics.
    """
    return Stats
#%%
analyse2([3, 7, 1, 9], 4, _prediction=True)
#%%
# expected result:
# Prediction(
#     Stats_mean_value=5.0,
#     Stats_above_threshold=[7.0, 9.0]
# )

# ✅ NOW WORKING! Fixed by adding return type annotation -> Stats

analyse2([3, 7, 1, 9], 4)
# expected result:
# Stats(mean_value=5.0, above_threshold=[7.0, 9.0])

# ✅ NOW WORKING! Fixed by adding return type annotation -> Stats

#%%
#==============================================
# Syntax 1.2: Using NamedTuple
#==============================================

@fd.ChainOfThought
def analyse3(numbers: List[float], # values to summarise and I have a lot to say about this 
            threshold: float # split point
            ) -> Stats:
    """
    Analyze numbers and return structured statistics.
    """
    class Stats(NamedTuple): mean_value: float; above_threshold: List[float]
    return Stats
#%%
analyse3([3, 7, 1, 9], 4)
#%%
analyse3([3, 7, 1, 9], 4, _prediction=True)

#%%
@fd.ChainOfThought
def analyse4(
    nums: List[float],  # list of numbers
    threshold: float   # split point
) -> tuple[float, List[float]]:
    """
    Analyze numbers and return structured statistics.
    """
    mean = "The average"
    above = "Numbers above threshold"
    return mean, above                       # body never runs
#%%
analyse4([3, 7, 1, 9], 4)
#%%
analyse4([3, 7, 1, 9], 4, _prediction=True)






# %%
