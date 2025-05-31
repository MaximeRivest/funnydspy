# FunkyDSpy Examples

This directory contains several examples demonstrating different features and use cases of the `funkydspy` library.

## Example Files

### 1. `example_basic_rag.py` - Basic RAG Example
**Purpose**: Demonstrates the simplest usage pattern with string output
- Shows basic RAG (Retrieval-Augmented Generation) functionality
- Uses chain-of-thought reasoning (`fd.cot`)
- Simple context + question → answer workflow
- **Run**: `python example_basic_rag.py`

### 2. `example_simple_number.py` - Simple Number Extraction
**Purpose**: Minimal example for basic question answering
- Demonstrates number extraction from text
- Minimal setup and configuration
- Good starting point for beginners
- **Run**: `python example_simple_number.py`

### 3. `example_analysis_multiple_returns.py` - Multiple Return Values
**Purpose**: Shows functions that return multiple values without structured types
- Demonstrates statistical analysis with multiple outputs
- Shows how funkydspy handles tuple returns
- Uses `fd.predict` for direct prediction
- **Run**: `python example_analysis_multiple_returns.py`

### 4. `example_namedtuple_analysis.py` - Structured Output with NamedTuple
**Purpose**: Advanced example showing structured output handling
- Uses `NamedTuple` for type-safe, structured returns
- Compares different DSPy modules (`fd.cot` vs `fd.predict`)
- Shows how to handle complex data types (lists, structured data)
- **Run**: `python example_namedtuple_analysis.py`

## Quick Start

Each example is self-contained and can be run independently:

```bash
# Try the basic RAG example
python example_basic_rag.py

# Compare different reasoning approaches
python example_namedtuple_analysis.py
```

## Key Concepts Demonstrated

- **@fd.funky decorator**: Transform regular functions into DSPy modules
- **Pipeline syntax**: Use `|` operator for chaining operations
- **Multiple reasoning modes**: Compare `fd.cot` (chain-of-thought) vs `fd.predict`
- **Type handling**: From simple strings to complex structured outputs
- **Docstring integration**: How function documentation guides LLM behavior

## Original Combined File

The original `use_fd.py` contains all examples in one file for comparison purposes. 