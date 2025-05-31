#%%
import funkydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

#%%
@fd.funky
def rag(context: str, question: str): return answer

@fd.funky  
def foo(x: int): return y

#%%
("The number is 1", "What is the number?") | rag | fd.cot  # Gets ChainOfThought for rag signature
#%%
(9.11,)       | foo | fd.cot  # Gets separate ChainOfThought for foo signature
#%%
# %%
