# %%
import funnydspy as fd
import dspy
import attachments as att

#%%
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano'))

# %%
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
res = att.attach(f"{url}[select: p]") | att.processors.webpage_to_llm | att.split.paragraphs



@fd.ChainOfThought
def classify(query: str, context: str) -> str: return answer

answer = rag("What is the capital of France?", "France is a country in Europe.")
print(answer)
# %%

from typing import Literal
emotions = ['happy', 'sad', 'angry', 'excited', 'fearful', 'disgusted', 'surprised']

@fd.ChainOfThought  
def classifier1(sentence: str) -> Literal[*emotions]:
    return emotion

# %%
classifier1("I am happy")
classifier1.module
# %%
from typing import Literal

@fd.ChainOfThought  
def classifier2(sentence: str) -> Literal['happy', 'sad', 'angry', 'excited', 'fearful', 'disgusted', 'surprised']:
    return emotion

classifier2("I am happy")
classifier2.module
# %%
