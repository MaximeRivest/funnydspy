#!/usr/bin/env python3

# %%
from typing import NamedTuple, List, Literal
from dataclasses import dataclass
import funnydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.ChainOfThought
def base_writer(parent_heading: list[str],
                content_chunk):
    """Turn into comprehensive but terse Markdown section.
    Use only headings deeper than the parent_headings."""
    return subsection

@fd.Predict
def summarizer(parent_headings: list[str], chunk): return gist

@fd.ChainOfThought
def heading_writer(parent_headings: list[str], chunk_gists: list[str]): return content_headings

def structure_and_summarize(parent_headings: list[str],chunks: list[str]):
    """Structure and summarize the content chunks."""
    if len(chunks) <= 4 or len(parent_headings) >= 3: return base_writer(parent_headings, chunks)
    
    parallel_executor = dspy.Parallel()
    exec_pairs = [(summarizer.module, {'parent_headings': parent_headings, 'chunk': c}) for c in chunks]
    chunk_gists_predictions = parallel_executor.forward(exec_pairs)
    
    # Extract the actual gist values from the predictions
    chunk_gists = [pred.gist if hasattr(pred, 'gist') else str(pred) for pred in chunk_gists_predictions]
    
    headers_result = heading_writer(parent_headings, chunk_gists)
    
    # Extract headers from the result - it might be a Prediction object or a string/list
    if hasattr(headers_result, 'content_headings'):
        headers = headers_result.content_headings
    elif isinstance(headers_result, list):
        headers = headers_result
    elif isinstance(headers_result, str):
        # If it's a string, try to split it or treat as single header
        headers = [h.strip() for h in headers_result.split(',') if h.strip()]
    else:
        headers = [str(headers_result)]

    @fd.ChainOfThought
    def classifier(parent_headings: list[str], chunk) -> str:
        """Classify the content headings into a category."""
        return topic

    exec_pairs = [(classifier.module, {'parent_headings': parent_headings, 'chunk': c}) for c in chunks]
    topics_predictions = parallel_executor.forward(exec_pairs)
    
    # Extract the actual topic values from the predictions
    topics = [pred.topic if hasattr(pred, 'topic') else str(pred) for pred in topics_predictions]
    
    # group chunks by their classified topics (use all unique topics, not just headers)
    all_topics = set(topics)  # Get all unique topics from classification
    sections = {topic: [] for topic in all_topics}
    
    for topic, chunk in zip(topics, chunks):
        sections[topic].append(chunk)
    
    # recursively process each section as a collection of chunks
    prefix = '#' * (len(parent_headings) + 1) + ' '
    exec_pairs = [(structure_and_summarize, {'parent_headings': parent_headings + [prefix + topic], 'chunks': section_chunks})
                  for topic, section_chunks in sections.items() if section_chunks]
    summarized_sections_predictions = parallel_executor.forward(exec_pairs)
    
    # Extract the actual content from the predictions
    summarized_sections = [pred if isinstance(pred, str) else str(pred) for pred in summarized_sections_predictions]
    
    # combine the summarized sections into a single string
    return '\n\n'.join([parent_headings[-1]] + summarized_sections)

#%%
# %%
import attachments as att
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
res = att.attach(f"{url}[select: p]") | att.processors.webpage_to_llm | att.split.paragraphs

# %%
content = structure_and_summarize([att.Attachments(url+"[select: title]").text],
                                   [t.text for t in res[:10]])

# # %%
# print(content)

# # https://en.wikipedia.org/wiki/Artificial_intelligence

# # Artificial intelligence - Wikipedia


# ## File Info

# - **Content Type**: text/html; charset=UTF-8
# - **Status Code**: 200



# ### Overview of Artificial Intelligence

# History of AI Development: From Rule-Based Reasoning to Probabilistic Methods

# Limitations of AI Reasoning and Human Problem-Solving Strategies

# Knowledge Representation and Its Applications

# Traits and Capabilities of AI Systems

# ## Knowledge Bases and Ontologies in AI

# A knowledge base is a body of knowledge represented in a form that can be used by a program. An ontology is the set of objects, relations, concepts, and properties used by a particular domain of knowledge.[23] Knowledge bases need to represent things such as objects, properties, categories, and relations between objects;[24] situations, events, states, and time;[25] causes and effects;[26] knowledge about knowledge (what we know about what other people know);[27] default reasoning (things that humans assume are true until they are told differently and will remain true even when other facts are changing);[28] and many other aspects and domains of knowledge.

# ## Common Applications of Artificial Intelligence