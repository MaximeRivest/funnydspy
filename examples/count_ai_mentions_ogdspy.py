#%% [markdown]
# # DSPy AI Mentions Counter: 50-Line Starter Guide
# 
# This notebook demonstrates how to use DSPy to optimize a simple AI task: counting mentions of "Artificial Intelligence" or "AI" in text paragraphs.
# 
# ## Overview
# We'll:
# 1. Define a DSPy signature for counting AI mentions
# 2. Fetch data from Wikipedia
# 3. Create a training dataset using a stronger model (Sonnet)
# 4. Optimize a weaker model (Flash) to match the stronger model's performance

#%%
import funnydspy as fd
import dspy
from attachments import Attachments
from datar import f
import datar.base as b
from datar.tibble import tibble
from datar.dplyr import mutate, summarise, n

dspy.configure(lm=dspy.LM('gemini/gemini-2.0-flash-lite', temperature = 1.0, max_tokens = 6000))

#%% [markdown]
# ## Step 1: Define the AI Task Signature
# 
# In DSPy, we define the task using a Signature class instead of writing prompts manually.

#%%
# This define the signature of the AI function. The replaces prompts.
class count_ai_occurrences(dspy.Signature):
    """Count the number times the word 'Artificial Intelligence'
    or 'AI' appears in the paragraph"""
    paragraph: str = dspy.InputField(desc = "The paragraph to count the AI mentions in")
    ai_occurrences_count: int = dspy.OutputField(desc = "The number of times the word 'Artificial Intelligence' or 'AI' appears in the paragraph")

dspy_module = dspy.Predict(count_ai_occurrences)
def count_ai_occurrences_f(paragraph):
    return dspy_module(paragraph=paragraph).ai_occurrences_count

#%% [markdown]
# ## Step 2: Fetch Training Data
# 
# We'll use the Attachments library to scrape Wikipedia and get paragraphs about AI.

#%%
# This fetches the AI wikipedia page and splits it into paragraphs
attachments_dsl = "[images: false][select: p,title,h1,h2,h3,h4,h5,h6][split: paragraphs]"
a = Attachments("https://en.wikipedia.org/wiki/Artificial_intelligence" + attachments_dsl) 

# This creates a dataframe with the paragraphs and the flash response
df = (tibble(paragraphs = [p.text for p in a[:10]]) >>
    mutate(flash_response = f.paragraphs.apply(count_ai_occurrences_f))
    )

#%% [markdown]
# ## Step 3: Create Gold Standard Labels
# 
# We use a stronger model (Claude Sonnet) to create the "correct" answers for our training set.

#%%
# This creates a column with the sonnet response, it will be used as the goldset
with dspy.context(lm=dspy.LM('anthropic/claude-sonnet-4-20250514')):
    df_with_goldset_col = mutate(df, resp_sonnet = f.paragraphs.apply(count_ai_occurrences_f))

#%% [markdown]
# ## Step 4: Prepare Training Dataset
# 
# Convert our data into DSPy's expected format for training.

#%%
# Reshape the data into a format that can be used for training
trainset = []
for r in df_with_goldset_col.to_dict(orient='records'):
    trainset.append(dspy.Example(
        paragraph=r['paragraphs'],           # this is the input
        ai_occurrences_count=r["resp_sonnet"]). # this is the target
        with_inputs('paragraph'))            # this is needed (not sure why)

#%% [markdown]
# ## Step 5: Optimize the Model
# 
# Use DSPy's optimizer to improve our weaker model's performance by learning from the stronger model.

#%%
# Define the metric for the optimizer
def exact_match(x, y, trace=None): return x.ai_occurrences_count == y.ai_occurrences_count

# Compile the optimizer
optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=exact_match, num_threads=24)
optimized_dspy_module = optimizer.compile(dspy_module, trainset=trainset)

def count_ai_occurrences_opt(paragraph):
    return optimized_dspy_module(paragraph=paragraph).ai_occurrences_count

#%% [markdown]
# ## Step 6: Evaluate Performance
# 
# Compare the performance before and after optimization to see the improvement.

#%%
# Calculate the performance of the optimized model
final_performance = (df_with_goldset_col >>
    mutate(
        #Applies flash to every row with the optimized prompt
        resp_flash_opt = f.paragraphs.apply(count_ai_occurrences_opt)) >>
    mutate(
        # Add 2 columns with 0 or 1 if the flash response is equal to the sonnet response
        flash_eq_sonnet = f.resp_sonnet == f.flash_response, #Compare flash with sonnet
        flash_opt_eq_sonnet = f.resp_flash_opt == f.resp_sonnet #Compare opt flash with sonnet
        ) >> 
    summarise(
        # Sum the number of rows where the flash response is equal to the sonnet response
        flashlight_before_opt = b.sum(f.flash_eq_sonnet)/n() *100, #n() is the number of rows in df
        # Sum the number of rows where the opt flash response is equal to the sonnet response
        flashlight_after_opt = b.sum(f.flash_opt_eq_sonnet)/n() *100 #n() is the number of rows in df
        ) >>
    mutate(precision_increase=f.flashlight_after_opt-f.flashlight_before_opt )
    )

#%% [markdown]
# ## Results
# 
# Let's see how much we improved the weaker model's performance! 🚀

#%%
f"The precision increased by {final_performance['precision_increase'].values[0]:.2f}% 🔥"
# %%
