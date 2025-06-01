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
    
    chunk_gists = dspy.Parallel(summarizer.module)(
              [{'parent_headings': parent_headings, 'chunk': c} for c in chunks])
    
    headers = heading_writer(parent_headings, chunk_gists)

    @fd.ChainOfThought
    def classifier(parent_headings: list[str], chunk) -> str:
        """Classify the content headings into a category."""
        return topic

    topics = dspy.Parallel(classifier.module)(
              [{'parent_headings': parent_headings, 'chunk': c} for c in chunks])
    
    # group chunks by into their sections
    sections = {topic: [] for topic in headers}
    for topic, chunk in zip(topics, chunks):
        sections[topic].append(chunk)
    
    # recursively process each section as a collection of chunks
    prefix = '#' * (len(parent_headings) + 1) + ' '
    summarized_sections = dspy.Parallel(structure_and_summarize)(
        [{'parent_headings': parent_headings + [prefix + topic], 'chunks': section_chunks}
        for topic, section_chunks in sections.items() if section_chunks]
    )
    
    # combine the summarized sections into a single string
    return '\n\n'.join([parent_headings[-1]] + summarized_sections)

#%%
#Usage
content = structure_and_summarize(['Welcome to funnydspy'], ['This is a test', 'This is a test'])
