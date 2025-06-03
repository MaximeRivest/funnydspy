# %% [markdown]
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

# %%

from datar.dplyr import mutate, summarise, n
from datar.tibble import tibble
import datar.base as b
from datar import f
from attachments import Attachments
import dspy
import funnydspy as fd
import os
os.environ["DSPY_CACHEDIR"] = os.path.join(
    os.path.dirname(__file__), ".litellm_cache")


lm = dspy.LM('gemini/gemini-2.0-flash-lite', temperature=1.0, max_tokens=6000)
dspy.configure(lm=lm)


# %% [markdown]
# ## Step 1: Define the AI Task Signature
#
# In DSPy, we define the task using a Signature class instead of writing prompts manually.
dspy.ChainOfThought()
import dspy
lm = dspy.LM('gemini/gemini-2.0-flash-lite', temperature=1.0, max_tokens=6000)
dspy.configure(lm=lm)

# Define the AI task signature
ai_counter = dspy.Predict("paragraph -> ai_occurrences_count: float")

# Run the AI task
ai_counter(paragraph="This is about Deep Neural Networks")

# Prompt produced by dspy.Predict
[
    {
    'role': 'system',
    'content': """
            Your input fields are:
            1. `paragraph` (str)
            Your output fields are:
            1. `ai_occurrences_count` (float)
            All interactions will be structured in the following way, with the appropriate values filled in.

            [[ ## paragraph ## ]]
            {paragraph}

            [[ ## ai_occurrences_count ## ]]
            # note: the value you produce must be a single float value
            {ai_occurrences_count}

            [[ ## completed ## ]]
            In adhering to this structure, your objective is:
            Given the fields `paragraph`, produce the fields `ai_occurrences_count`.
        """},
    {
    'role': 'user',
    'content': """
        [[ ## paragraph ## ]]
        This is about Deep Neural Networks

        Respond with the corresponding output fields, starting with the field `[[ ## ai_occurrences_count ## ]]` 
        (must be formatted as a valid Python float), and then ending with the marker for `[[ ## completed ## ]]`.

        """
    }
]


# %%



class count_ai_occurrences(dspy.Signature):
    """Count the number times the word 'Artificial Intelligence'
    or 'AI' or any other reference to AI or AI-related terms appears in the paragraph"""
    paragraph: str= dspy.InputField(desc = "The paragraph to count the AI mentions in")
    ai_occurrences_count: int= dspy.OutputField(desc = "The number of times the word 'Artificial Intelligence' or 'AI' appears in the paragraph")

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
#%%
# This creates a dataframe with the paragraphs and the flash response
df = (tibble(paragraphs = [p.text for p in a[:20]]) >>
    mutate(flash_response= f.paragraphs.apply(count_ai_occurrences_f))
    )

#%% [markdown]
# ## Step 3: Create Gold Standard Labels
#
# We use a stronger model (Claude Sonnet) to create the "correct" answers for our training set.

#%%
# This creates a column with the sonnet response, it will be used as the goldset
with dspy.context(lm=dspy.LM('anthropic/claude-sonnet-4-20250514')):
    df_with_goldset_col= mutate(df, resp_sonnet = f.paragraphs.apply(count_ai_occurrences_f))


(mutate(df_with_goldset_col, exact_match = f.resp_sonnet == f.flash_response) >>
    summarise(baseline_precision = b.sum(f.exact_match)/n() * 100))

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
        ai_occurrences_count=r["resp_sonnet"]).  # this is the target
       with_inputs('paragraph'))            # this is needed (not sure why)

#%% [markdown]
# ## Step 5: Optimize the Model
#
# Use DSPy's optimizer to improve our weaker model's performance by learning from the stronger model.

        #%%
# Define the metric for the optimizer
def exact_match(x, y, trace=None): return x.ai_occurrences_count == y.ai_occurrences_count

# Compile the optimizer
optimizer = dspy.MIPROv2(metric=exact_match, teacher_settings=dspy.LM('anthropic/claude-sonnet-4-20250514'))
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
        # Applies flash to every row with the optimized prompt
        resp_flash_opt= f.paragraphs.apply(count_ai_occurrences_opt)) >>
    mutate(
        # Add 2 columns with 0 or 1 if the flash response is equal to the sonnet response
        flash_eq_sonnet = f.resp_sonnet == f.flash_response,  # Compare flash with sonnet
        flash_opt_eq_sonnet = f.resp_flash_opt == f.resp_sonnet  # Compare opt flash with sonnet
        ) >>
    summarise(
        # Sum the number of rows where the flash response is equal to the sonnet response
        flashlite_before_opt = b.sum(f.flash_eq_sonnet)/n() * 100, #n() is the number of rows in df
        # Sum the number of rows where the opt flash response is equal to the sonnet response
        flashlite_after_opt = b.sum(f.flash_opt_eq_sonnet)/n() * 100 #n() is the number of rows in df
    ) >>
    mutate(precision_increase=f.flashlite_after_opt-f.flashlite_before_opt)
    )

#%% [markdown]
# ## Results
#
# Let's see how much we improved the weaker model's performance! 🚀

#%%
f"The precision increased by {final_performance['precision_increase'].values[0]:.2f}% 🔥"
# %%
