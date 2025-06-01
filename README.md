# FunnyDSPy [Experimental 🧪]

**Vanilla-Python ergonomics on top of DSPy**

I love DSPy, but I could not wrap my head around the fact that I have to write such a function-y Class which I could not use it as a normal Python function, with a 'normal' return value. So I wrote this library as an experiment to see if I could make it easier to use DSPy (for me!).

This library is a thin wrapper around DSPy that allows you to write plain functions and dataclasses and get DSPy modules *and* normal Python return values automatically. If you need the original DSPy `Prediction` (for optimization or loss computation), just call the function with `_prediction=True`.

## Example 1:
```python
import funnydspy as fd
import dspy

# Configure your language model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano'))#, api_key='YOUR_OPENAI_API_KEY'))

@fd.ChainOfThought
def rag(query: str, context: str) -> str: return answer

# Get Python objects directly
answer = rag("What is the capital of France?", "France is a country in Europe.")
# → "The capital of France is Paris."

# Get DSPy Prediction for optimization
pred = rag("What is the capital of France?", "France is a country in Europe.", _prediction=True)
# → dspy.Prediction(reasoning="...", answer="The capital of France is Paris.")
```

## Example 2:
```python
import funnydspy as fd
import dspy
from typing import NamedTuple

# Configure your language model
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano'))

@fd.ChainOfThought
def analyze(numbers: list[float], threshold: float) -> tuple[float, list[float]]:
    """Analyze numbers and return statistics."""
    class Stats(NamedTuple):
        mean: float # The average of the numbers
        above: list[float] # Numbers above the threshold
    return Stats

# Get Python objects directly
mean_val, above_vals = analyze([1, 5, 3, 8, 2], 4.0)
# → (4.0, [5.0, 8.0])

# Get DSPy Prediction for optimization
pred = analyze([1, 5, 3, 8, 2], 4.0, _prediction=True)
# → dspy.Prediction(reasoning="...", mean=4.0, above=[5.0, 8.0])
```

## ✨ Features

- **🐍 Pythonic**: Write normal Python functions with type hints
- **📦 Structured Returns**: Use dataclasses, NamedTuples, or tuples for complex outputs
- **🔄 DSPy Compatible**: Seamlessly integrates with DSPy optimization and chaining
- **📝 Smart Documentation**: Extracts descriptions from docstrings and inline comments
- **🎯 Type Safe**: Automatic type conversion between LM strings and Python types
- **⚡ Multiple Modules**: Support for Predict, ChainOfThought, ReAct, and custom modules

## 📦 Installation

```bash
pip install funnydspy
```

## 🚀 Quick Start

### Basic Usage with Dataclasses

```python
from dataclasses import dataclass
from typing import List
import funnydspy as fd
import dspy

dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano'))

@dataclass
class Stats:
    mean_value: float      # The average of all numbers
    above_threshold: List[float]  # Values greater than threshold

@fd.Predict
def analyze_data(numbers: List[float], threshold: float) -> Stats:
    """Analyze a list of numbers and return statistics."""
    return Stats

# Use it like a normal Python function
result = analyze_data([1, 5, 3, 8, 2], 4.0)
print(result.mean_value)        # 3.8
print(result.above_threshold)   # [5.0, 8.0]
```

### Tuple Returns with Variable Names

```python
@fd.ChainOfThought
def summarize_text(text: str) -> tuple[str, int, List[str]]:
    """Summarize text and extract key information."""
    summary = "A concise summary of the text"
    word_count = "Total number of words"
    key_points = "List of main points"
    return summary, word_count, key_points

summary, count, points = summarize_text("""Modules help you describe AI behavior as code, not strings.
To build reliable AI systems, you must iterate fast. But maintaining prompts makes that hard: it forces you to tinker with strings or data every time you change your LM, metrics, or pipeline. Having built over a dozen best-in-class compound LM systems since 2020, we learned this the hard way—and so built DSPy to decouple AI system design from messy incidental choices about specific LMs or prompting strategies.""")
```

### Using Different DSPy Modules

```python
# Chain of Thought reasoning
@fd.ChainOfThought
def complex_reasoning(problem: str) -> str:
    """Solve a complex problem step by step."""
    return solution

# Basic prediction
@fd.Predict
def simple_task(input_text: str) -> str:
    """Perform a simple text transformation."""
    return output
```


### Working with Optimizers

```python
# Access the underlying DSPy module for optimization
optimizer = dspy.BootstrapFewShot(metric=your_metric)
compiled_analyze = optimizer.compile(analyze_data.module, trainset=your_data)

# Wrap the optimized module back into a Pythonic interface
analyze_optimized = fd.funnier(compiled_analyze)

# Use the optimized version with the same interface
result = analyze_optimized([1, 5, 3, 8, 2], 4.0)
```

### ⚡ Elegant Parallel Execution

FunnyDSPy makes DSPy's parallel execution much more elegant and Pythonic with two approaches:

#### 1. `fd.parallel()` - Direct parallel execution
Works only with FunnyDSPy decorated functions:

```python
@fd.ChainOfThought
def analyze_text(text: str, context: str) -> str:
    """Analyze text in given context."""
    return analysis

# OLD VERBOSE DSPy SYNTAX:
# parallel = dspy.Parallel()
# pairs = [(analyze_text.module, {'text': t, 'context': ctx}) for t in texts]
# results = [pred.analysis for pred in parallel.forward(pairs)]

# ✨ NEW CLEAN FUNNYDSPY SYNTAX:
results = fd.parallel(analyze_text, [
    {'text': t, 'context': ctx} for t in texts
])
```

#### 2. `fd.parallelize()` - DSPy-style parallelization (recommended)
Works with any function, giving you the same experience as standard DSPy:

```python
@fd.ChainOfThought
def analyze_text(text: str, context: str) -> str:
    return analysis

# Create a parallelizable version
parallel_analyze = fd.parallelize(analyze_text)

# Use it like in standard DSPy
results = parallel_analyze([
    {'text': t, 'context': ctx} for t in texts
])
```

**Key advantage**: `fd.parallelize()` works with **any function**, including regular Python functions:

```python
def regular_function(x: int, y: int) -> int:
    return x + y

# This works! (executes sequentially for non-DSPy functions)
parallel_func = fd.parallelize(regular_function)
results = parallel_func([{'x': 1, 'y': 2}, {'x': 3, 'y': 4}])  # [3, 7]
```

#### Real-world recursive example (just like standard DSPy):
```python
def structure_and_summarize(parent_headings: List[str], chunks: List[str]) -> str:
    # ... base case ...
    
    # Parallelize DSPy functions
    produce_gist = fd.parallelize(gist_producer)
    chunk_gists = produce_gist([{'parent_headings': parent_headings, 'chunk': c} for c in chunks])
    
    # Even parallelize the recursive function itself!
    parallel_structure = fd.parallelize(structure_and_summarize)
    summarized_sections = parallel_structure([
        {'parent_headings': parent_headings + [prefix + topic], 'chunks': section_chunks}
        for topic, section_chunks in sections.items() if section_chunks
    ])
    
    return "\n\n".join([parent_headings[-1]] + summarized_sections)
```

#### Summary:
- **`fd.parallel(func, inputs)`**: Direct parallel execution (FunnyDSPy functions only)
- **`fd.parallelize(func)`**: Creates parallelizable version (any function, DSPy-style API)

## 📚 Documentation

### Decorators

- `@fd.Predict` - Basic prediction module
- `@fd.ChainOfThought` - Chain of thought reasoning

### Return Types

FunnyDSPy supports various return type patterns:

1. **Dataclasses**: Structured data with field descriptions
2. **NamedTuples**: Lightweight structured returns
3. **Tuples**: Simple multiple returns with automatic field naming
4. **Primitives**: Single values (str, int, float, etc.)

### Type Conversion

FunnyDSPy automatically handles conversion between LM string outputs and Python types:

- `str` → `str` (passthrough)
- `int` → `int` (parsed)
- `float` → `float` (parsed)
- `bool` → `bool` (true/false/yes/no/1/0)
- `List[T]` → `List[T]` (JSON or comma-separated)
- `Dict[K, V]` → `Dict[K, V]` (JSON parsed)

### Documentation Extraction

FunnyDSPy extracts field descriptions from multiple sources:

```python
@dataclass
class Result:
    """Result container."""
    value: float  # The computed value
    status: str   # Processing status

@fd.Predict
def process(
    data: List[float],  # Input data to process
    mode: str          # Processing mode
) -> Result:
    """
    Process data and return results.
    
    Parameters
    ----------
    data: The input dataset
    mode: How to process the data
    
    Returns
    -------
    Result.value: The final computed value
    Result.status: Success or error status
    """
    return Result
```

## 🔧 Advanced Usage

### Custom DSPy Modules

```python
# Register custom DSPy modules
class CustomModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.Predict(signature)
    
    def forward(self, **kwargs):
        return self.predictor(**kwargs)

fd.register(CustomModule, alias="custom")

@fd.custom
def my_function(input_text: str) -> str:
    return output
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of the excellent [DSPy](https://github.com/stanfordnlp/dspy) framework
- Inspired by the need for more Pythonic LM programming interfaces