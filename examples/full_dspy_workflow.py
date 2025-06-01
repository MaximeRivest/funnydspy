# %%
import funnydspy as fd
import dspy
from attachments import Attachments
from attachments import __version__
from datar import f
from datar.tibble import tibble
from datar.dplyr import mutate, select, arrange, filter, group_by, summarise, n
print(__version__)

# %%
attachments_dsl = "[images: false][select: p,title,h1,h2,h3,h4,h5,h6][split: paragraphs]"
a = Attachments("https://en.wikipedia.org/wiki/Artificial_intelligence" + attachments_dsl) 

#%%
@fd.Predict
def count_ai_words(paragraph) -> float:
    """Count the number of Artificial Intelligence or AI words in the paragraph"""
    return ai_frequency

#%%
from typing import Literal
@fd.Predict
def classify_paragraph_subject(paragraph) -> Literal['Yes', 'No']:
    return is_main_subject_deep_learning
#%%

#%%
#dspy.configure(lm=dspy.LM('anthropic/claude-sonnet-4-20250514'))
#dspy.configure(lm=dspy.LM('gemini/gemma-3-27b-it'))
dspy.configure(lm=dspy.LM('gemini/gemini-2.0-flash-lite'))
#%%

count_ai_words(a[1].text)

#%%


res = classify_paragraph_subject(a[0].text, _prediction=True)
print(res)
#%%
res

#%%


df = tibble(paragraphs = [p.text for p in a[:20]])
df

#%%
holder = []
for p in a[:10]:
    holder.append(classify_paragraph_subject(p.text, _prediction=True))

holder

#%%
df1 = df >> mutate(is_main_subject_deep_learning = classify_paragraph_subject(f.paragraphs))
df1

#%%
with dspy.context(lm=dspy.LM('gemini/gemma-3n-e4b-it')):
    df2 = df1 >> mutate(resp_gemma = classify_paragraph_subject(f.paragraphs))

df2

#%%
# use sonnet 4 to classify the paragraphs store in Ibis + duckdb

# make is easy to evaluate base flash light on mimicking sonnet 4

# optimize the same classification based on flash light

# reevaluate easily and dplyr y

# make the program funnier and apply to every rows with a mutate in Ibis.

