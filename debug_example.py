#!/usr/bin/env python3

import sys
sys.path.insert(0, '..')

import funnydspy as fd
import dspy
from typing import List, Literal

# Configure DSPy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.Predict  
def gist_producer(parent_headings: List[str], chunk: str) -> str: return gist

@fd.ChainOfThought
def header_producer(parent_headings: List[str], chunk_gists: List[str]) -> List[str]: return content_headings

print("Testing gist_producer in parallel...")
parent_headings = ['Test']
chunks = ['chunk1', 'chunk2']
chunk_gists = fd.parallel(gist_producer, [
    {'parent_headings': parent_headings, 'chunk': c} for c in chunks
])
print(f"chunk_gists: {chunk_gists}")
print(f"types: {[type(x) for x in chunk_gists]}")

print("\nTesting header_producer...")
headers = header_producer(parent_headings, chunk_gists)
print(f"headers: {headers}")
print(f"type: {type(headers)}")

# Test the classifier creation
print(f"\nCreating classifier with headers: {headers}")

@fd.ChainOfThought  
def classifier(parent_headings: List[str], chunk: str) -> Literal[*headers]: return topic

print("Testing classifier in parallel...")
topics = fd.parallel(classifier, [
    {'parent_headings': parent_headings, 'chunk': c} for c in chunks
])
print(f"topics: {topics}")
print(f"types: {[type(x) for x in topics]}")

# Test if topics can be used as dictionary keys
print(f"\nTesting dictionary usage...")
sections = {topic: [] for topic in headers}
print(f"sections dict created: {sections}")

try:
    for topic, chunk in zip(topics, chunks):
        print(f"Checking if {topic} (type: {type(topic)}) is in sections...")
        if topic in sections:
            sections[topic].append(chunk)
            print(f"Added {chunk} to {topic}")
        else:
            print(f"Topic {topic} not in sections keys: {list(sections.keys())}")
    print(f"Final sections: {sections}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 