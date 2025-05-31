"""
Simple Number Extraction Example
================================
A minimal example showing basic question answering with number extraction.
"""

import funkydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano'))

@fd.funky
def rag(context: str, question: str) -> str:
    """Extract information from context to answer questions."""
    return answer

("The number is 1", "What is the number?") | rag | fd.cot
