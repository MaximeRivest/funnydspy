"""
FunkyDSpy Examples - Combined File
=================================

NOTE: This file contains multiple examples in one place for comparison.
Individual examples have been split into separate files for easier learning:

- example_basic_rag.py           : Basic RAG with string output
- example_simple_number.py       : Simple number extraction  
- example_analysis_multiple_returns.py : Multiple return values
- example_namedtuple_analysis.py : Structured output with NamedTuple

See README_examples.md for detailed descriptions and usage instructions.
"""

#%%
import funkydspy as fd           # ‹– everything lives in this tiny helper
import dspy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

#%%
@fd.funky                        # ①   declare with def-syntax
def rag(context: str, question: str) -> str:
    return answer                # body is *never executed* – it's just a stub
#%%
result = (
    "Lee scored twice for the U's last season.",
    "How many goals did Lee score?"
) | rag | fd.cot                 # ②   no extra parentheses, no signature arg
#%%
print(result)             # ← works

#Prediction(
#    reasoning='The statement indicates that Lee scored twice last season, which means he scored 2 goals during that period. The phrase "scored twice" directly refers to the number of goals, so the total goals scored by Lee is 2.',
#    answer='2'
#)



#%%

@fd.funky
def analyse(numbers: list[float], threshold: float):
    """
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
    return mean_value, above_threshold
#%%
data      = [3.5, 7.2, 1.8, 9.0]
cut_off   = 4.0

#%%
# choose any DSPy module you like; e.g. fd.predict
result = (data, cut_off) | analyse | fd.predict

print("Mean :", result.mean_value)
print("High :", result.above_threshold)

# %%
import funkydspy as fd
import dspy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano'))

@fd.funky
def rag(context: str, question: str) -> str:
    return answer

("The number is 1","What is the number?") | rag | fd.cot
# %%


# -------------------------------------------
# 1  declare with def + NamedTuple annotation
# -------------------------------------------
from typing import NamedTuple, List
import funkydspy as fd

class Stats(NamedTuple):
    mean_value       : float
    above_threshold  : List[float]   # list to show non-str output handling

@fd.funky
def analyse(numbers: List[float], threshold: float) -> Stats:
    """
    Returns
    -------
    mean_value      : float        average of *numbers*
    above_threshold : list[float]  values > threshold
    """
    return Stats    # body never executes

# %%
# -------------------------------------------
# 2  run the pipeline with any DSPy module
# -------------------------------------------
data    = [3.5, 7.2, 1.8, 9.0]
cut_off = 4.0

# Chain-of-Thought
res1 = (data, cut_off) | analyse | fd.cot
print("mean:",  res1.mean_value)
print("high:",  res1.above_threshold)

# or swap reasoning style with zero glue:
res2 = (data, cut_off) | analyse | fd.predict
print("predict →", res2.mean_value, res2.above_threshold)

# %%
