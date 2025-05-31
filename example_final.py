"""
NamedTuple Analysis Example
===========================
This example demonstrates using NamedTuple for structured output from funkydspy functions.
Shows how to define typed return values and use different DSPy modules (cot vs predict).
"""
# %%
from typing import NamedTuple, List
from dataclasses import dataclass
import fcspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

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
def analyse(numbers: List[float],
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
#%%
#==============================================
# Syntax 1.1: Using dataclase but with docment
#==============================================
@dataclass
class Stats:
    """Structured output for analysis results."""
    mean_value: float # values to summarise
    above_threshold: List[float] # values > threshold


@fd.funky
def analyse(numbers: List[float], # values to summarise
            threshold: float # split point
            ):
    """
    Analyze numbers and return structured statistics.
    """
    return Stats

# ==============================================
# Syntax 2
# Using tuple for multiple returns
# and using inline docment for  input
# description (when needed). For output description
# use string assignment. The variable names with be
# used as the variable names in the output.
# for no description, use "".
# The docstring is use for the Class description.
# ==============================================
@fd.funky
def analyse(
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
# this should work by default (using predict)
r = analyse([3, 7, 1, 9], 4)
print(r)
#%%
# Using Chain of Thought via pipeline:
result_cot = ([3, 7, 1, 9], 4) | analyse | fd.cot
print(result_cot)
#%%
# or in pipeline:
([3, 7, 1, 9], 4) | analyse | fd.cot

#%%
# ==============================================
@fd.funky
def analyse(
    nums: List[float],
    threshold: float
) -> tuple[float, List[float]]:
    return mean, above

#%%
# Using predict via pipeline:
analyse([3, 7, 1, 9], 4)

#%%
# Using predict via pipeline:
([3, 7, 1, 9], 4) | analyse | fd.predict

#%%
# Using Chain of Thought via pipeline:
([3, 7, 1, 9], 4) | analyse | fd.cot

# ==============================================
@fd.funky
def rag(context, question): return answer

rag("The number is 1", "What is the number?")

# %%



import fcspy as fd
import dspy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.funky
def analyse(
    nums: List[float],  # inline: list of numbers
    threshold: float   # inline: split point
) -> tuple[float, List[float]]:
    """
    Analyze numbers and return structured statistics.
    """
    mean = "The average"
    above = "Numbers above threshold"
    return mean, above                       # body never runs

analyse([3, 7, 1, 9], 4)
