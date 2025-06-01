#%%
from typing import List, Literal
import funnydspy as fd
import dspy

# Configure DSPy with OpenAI model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.ChainOfThought
def base_writer(parent_headings: List[str], content_chunks: List[str]) -> str:
    """Turn into comprehensive but terse Markdown section. Use only headings deeper than parent_headings."""
    return subsection

@fd.Predict  
def gist_producer(parent_headings: List[str], chunk: str) -> str: return gist

@fd.ChainOfThought
def header_producer(parent_headings: List[str], chunk_gists: List[str]) -> List[str]: 
    return content_headings

def structure_and_summarize(parent_headings: List[str], chunks: List[str]) -> str:
    # 1. Base Case: If the work left is small, just write the section
    if len(chunks) <= 3 or len(parent_headings) >= 2:
        return base_writer(parent_headings, chunks)
    
    # 2. Summarize each chunk in parallel using parallelize()
    chunk_gists = fd.parallelize(gist_producer)([
        {'parent_headings': parent_headings, 'chunk': c} for c in chunks
    ])
    
    # 3. Prepare next level of Table of Contents
    headers = header_producer(parent_headings, chunk_gists)
    print(headers)
    
    # 4. Create dynamic classifier with Literal headers and assign chunks
    @fd.ChainOfThought  
    def classifier(parent_headings: List[str], chunk: str) -> Literal[*headers]: return topic
    
    topics = fd.parallelize(classifier)([
        {'parent_headings': parent_headings, 'chunk': c} for c in chunks
    ])
    
    # 5. Group chunks into sections
    sections = {topic: [] for topic in headers}
    for topic, chunk in zip(topics, chunks):
        if topic in sections:  # Only add if topic is in our headers
            sections[topic].append(chunk)
    
    # 6. Recursively process each section in parallel using parallelize()
    prefix = "#" * (len(parent_headings) + 1) + " "
    parallel_structure_and_summarize = fd.parallelize(structure_and_summarize)
    summarized_sections = parallel_structure_and_summarize([
        {'parent_headings': parent_headings + [prefix + topic], 'chunks': section_chunks}
        for topic, section_chunks in sections.items() if section_chunks
    ])
    
    # 7. Collect sub-sections together
    return "\n\n".join([parent_headings[-1]] + summarized_sections)

# %%
import attachments as att

url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
res = att.attach(f"{url}[select: p]") | att.processors.webpage_to_llm | att.split.paragraphs

#%%
content = structure_and_summarize(
    [att.Attachments(url+"[select: title]").text],
    [t.text for t in res[:4]]
)
print(content)

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
# %%
