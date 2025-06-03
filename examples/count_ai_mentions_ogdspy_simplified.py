from datar.dplyr import mutate, summarise, n
from datar.tibble import tibble
import datar.base as b
from datar import f
from attachments import Attachments
import dspy

dspy.configure(lm=dspy.LM('gemini/gemini-2.0-flash-lite'))

#Setup dspy program
class count_ai_occurrences(dspy.Signature):
    """Count the number times the word 'Artificial Intelligence' or 'AI' or any other reference to AI or AI-related terms appears in the paragraph"""
    paragraph: str= dspy.InputField(desc = "The paragraph to count the AI mentions in")
    ai_occurrences_count: int= dspy.OutputField(desc = "The number of times the word 'Artificial Intelligence' or 'AI' appears in the paragraph")

dspy_module = dspy.Predict(count_ai_occurrences)

def count_ai_occurrences_f(paragraph):
    return dspy_module(paragraph=paragraph).ai_occurrences_count

# This fetches the AI wikipedia page and splits it into paragraphs
attachments_dsl = "[images: false][select: p,title,h1,h2,h3,h4,h5,h6][split: paragraphs]"
a = Attachments("https://en.wikipedia.org/wiki/Artificial_intelligence" + attachments_dsl)

# This creates a dataframe with the paragraphs and the flash response
df = (tibble(paragraphs = [p.text for p in a[:20]]) >>
    mutate(flash_response= f.paragraphs.apply(count_ai_occurrences_f)))

# This creates a column with the sonnet response, it will be used as the goldset
with dspy.context(lm=dspy.LM('anthropic/claude-sonnet-4-20250514')):
    df_with_goldset_col= mutate(df, resp_sonnet = f.paragraphs.apply(count_ai_occurrences_f))

#Just printing the baseline precision
(mutate(df_with_goldset_col, exact_match = f.resp_sonnet == f.flash_response) >>
    summarise(baseline_precision = b.sum(f.exact_match)/n() * 100))

# Reshape the data into a format that can be used for training
trainset = []
for r in df_with_goldset_col.to_dict(orient='records'):
    trainset.append(dspy.Example(
        paragraph=r['paragraphs'],           # this is the input
        ai_occurrences_count=r["resp_sonnet"]).  # this is the target
       with_inputs('paragraph'))            # this is needed (not sure why)

# Define the metric for the optimizer
def exact_match(x, y, trace=None): return x.ai_occurrences_count == y.ai_occurrences_count

# Compile the optimizer
optimizer = dspy.MIPROv2(metric=exact_match, teacher_settings=dspy.LM('anthropic/claude-sonnet-4-20250514'))
optimized_dspy_module = optimizer.compile(dspy_module, trainset=trainset)

def count_ai_occurrences_opt(paragraph):
    return optimized_dspy_module(paragraph=paragraph).ai_occurrences_count

# That's it with DSPy, you can use the optimized model like this:
count_ai_occurrences_opt("This is about Deep Neural Networks")

# Using Datar to calculate the performance of the optimized model
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

f"The precision increased by {final_performance['precision_increase'].values[0]:.2f}% 🔥"
