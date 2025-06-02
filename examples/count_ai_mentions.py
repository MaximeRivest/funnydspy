#%%
import funnydspy as fd
import dspy
from attachments import Attachments
from datar import f
import datar.base as b
from datar.tibble import tibble
from datar.dplyr import mutate, summarise, n

dspy.configure(lm=dspy.LM('gemini/gemini-2.0-flash-lite'))

# This define the signature of the AI function. The replaces prompts.
@fd.Predict
def count_flash_response(paragraph) -> float:
    """Count the number times the word 'Artificial Intelligence'
    or 'AI' appears in the paragraph"""
    return mention_frequency

# This fetches the AI wikipedia page and splits it into paragraphs
attachments_dsl = "[images: false][select: p,title,h1,h2,h3,h4,h5,h6][split: paragraphs]"
a = Attachments("https://en.wikipedia.org/wiki/Artificial_intelligence" + attachments_dsl) 

# This creates a dataframe with the paragraphs and the flash response
df = (tibble(paragraphs = [p.text for p in a[:20]]) >>
    mutate(flash_response = f.paragraphs.apply(count_flash_response))
    )

# This creates a column with the sonnet response, it will be used as the goldset
with dspy.context(lm=dspy.LM('anthropic/claude-sonnet-4-20250514')):
    df_with_goldset_col = mutate(df, resp_sonnet = f.paragraphs.apply(count_flash_response))

# Reshape the data into a format that can be used for training
trainset = []
for r in df_with_goldset_col.to_dict(orient='records'):
    trainset.append(dspy.Example(
        paragraph=r['paragraphs'],           # this is the input
        mention_frequency=r["resp_sonnet"]). # this is the target
        with_inputs('paragraph'))            # this is needed (not sure why)

# Define the metric for the optimizer
def exact_match(x, y, trace=None): return x.mention_frequency == y.mention_frequency

# Compile the optimizer
optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=exact_match, num_threads=24)
optimized = optimizer.compile(count_flash_response.module, trainset=trainset)
count_flash_response_opt = fd.funnier(optimized)

final_performance = (df_with_goldset_col >>
    mutate(
        #Applies flash to every row with the optimized prompt
        resp_flash_opt = f.paragraphs.apply(count_flash_response_opt)) >>
    mutate(
        # Add 2 columns with 0 or 1 if the flash response is equal to the sonnet response
        flash_eq_sonnet = f.resp_sonnet == f.flash_response, #Compare flash with sonnet
        flash_opt_eq_sonnet = f.resp_flash_opt == f.resp_sonnet #Compare opt flash with sonnet
        ) >> 
    summarise(
        # Sum the number of rows where the flash response is equal to the sonnet response
        flashlight_before_opt = b.sum(f.flash_eq_sonnet)/n(), #n() is the number of rows in df
        # Sum the number of rows where the opt flash response is equal to the sonnet response
        flashlight_after_opt = b.sum(f.flash_opt_eq_sonnet)/n()) #n() is the number of rows in df
    )

final_performance