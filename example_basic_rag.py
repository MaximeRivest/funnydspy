"""
Basic RAG Example
================
This example demonstrates the simplest usage of funkydspy with a RAG (Retrieval-Augmented Generation)
function that takes a context and question, returning a string answer.
"""
#%%
import funkydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.funky
def rag(context: str, question: str) -> str:
    """Simple RAG function that answers questions based on context."""
    return answer  # body is *never executed* – it's just a stub

#%%
result = (
    "Lee scored twice for the U's last season.",
    "How many goals did Lee score?"
) | rag | fd.cot  # Chain-of-Thought reasoning
#%%
print(result)

#%%
# Expected output format:
# Prediction(
#     reasoning='The statement indicates that Lee scored twice last season, which means he scored 2 goals during that period. The phrase "scored twice" directly refers to the number of goals, so the total goals scored by Lee is 2.',
#     answer='2'
# ) 