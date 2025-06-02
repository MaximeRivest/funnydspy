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
from attachments.dspy import Attachments

turn_webpage_screenshot_off = "[images: false]"
cssselect = "[select: p,title,h1,h2,h3,h4,h5,h6]"
a = Attachments("https://en.wikipedia.org/wiki/Artificial_intelligence" \
                + turn_webpage_screenshot_off + cssselect)

dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

# This defines the signature of the AI function. The replaces prompts.
@fd.Predict
def rag(question,context: Attachments): return response

rag("What are the first 10 words of 'main' text?", a)
#> 'The first 10 words of the \'main\' text are: "Artificial...'




import funnydspy as fd
import dspy
from attachments.dspy import Attachments

turn_webpage_screenshot_off = "[images: false]"
cssselect = "[select: p,title,h1,h2,h3,h4,h5,h6]"
a = Attachments("https://en.wikipedia.org/wiki/Artificial_intelligence" \
                + turn_webpage_screenshot_off + cssselect)

dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

# This defines the signature of the AI function. The replaces prompts.
@fd.Predict
def rag(question,context: Attachments): return response

rag("What are the first 10 words of 'main' text?", a)
#> 'The first 10 words of the \'main\' text are: "Artificial...'



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
#> Prediction(
#     Stats_mean_value=5.0,
#     Stats_above_threshold=[7.0, 9.0]
# )

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
from typing import Tuple


@fd.funky
def analyse10(numbers, threshold):
    """
    Analyze numbers and return structured statistics.

    Parameters
    ----------
    numbers: List[float] values to summarise
    threshold: float split point

    Returns
    -------
    mean_value: float average of numbers
    above_threshold: List[float] values > threshold
    """
    pass

analyse10([3, 7, 1, 9], 4)
#> Stats(mean_value=5.0, above_threshold=[7.0, 9.0])
analyse([3, 7, 1, 9], 4, _prediction=True)
#> Prediction(
#     Stats_mean_value=5.0,
#     Stats_above_threshold=[7.0, 9.0]
# )

#%%
#==============================================
# Syntax 1.1: Using dataclase but with docment
#==============================================
@dataclass
class Stats:
    mean_value: float # values to summarise
    above_threshold: List[float] # values > threshold


@fd.Predict
def analyse(numbers: List[float], # values to summarise and I have a lot to say about this 
            threshold: float # split point
            ) -> Stats:
    """
    Analyze numbers and return structured statistics.
    """
    return Stats

analyse([3, 7, 1, 9], 4)
#> Stats(mean_value=5.0, above_threshold=[7.0, 9.0])
analyse([3, 7, 1, 9], 4, _prediction=True)
#> Prediction(
#     Stats_mean_value=5.0,
#     Stats_above_threshold=[7.0, 9.0]
# )


# expected result:
# Stats(mean_value=5.0, above_threshold=[7.0, 9.0])

# ✅ NOW WORKING! Fixed by adding return type annotation -> Stats

#%%
#==============================================
# Syntax 1.2: Using NamedTuple
#==============================================

@fd.ChainOfThought
def analyse(numbers: List[float], # values to summarise and I have a lot to say about this 
            threshold: float # split point
            ) -> Stats:
    """
    Analyze numbers and return structured statistics.
    """
    class Stats(NamedTuple): mean_value: float; above_threshold: List[float]
    return Stats

analyse([3, 7, 1, 9], 4)
#> Stats(mean_value=5.0, above_threshold=[7.0, 9.0])
analyse([3, 7, 1, 9], 4, _prediction=True)
#> Prediction(
#     Stats_mean_value=5.0,
#     Stats_above_threshold=[7.0, 9.0]
# )

#%%
@fd.ChainOfThought
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

analyse([3, 7, 1, 9], 4)
#> Stats(mean=5.0, above=[7.0, 9.0])
analyse([3, 7, 1, 9], 4, _prediction=True)
#> Prediction(
#     reasoning='The list of numbers provide...'
#     Stats_mean_value=5.0,
#     Stats_above_threshold=[7.0, 9.0]
# )





# %%
