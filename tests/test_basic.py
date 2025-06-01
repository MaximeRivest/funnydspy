"""Basic tests for funnydspy functionality."""

import pytest
from dataclasses import dataclass
from typing import List
import funnydspy as fd
import dspy


def test_import():
    """Test that the package imports correctly."""
    assert hasattr(fd, 'funky')
    assert hasattr(fd, 'Predict')
    assert hasattr(fd, 'ChainOfThought')
    assert hasattr(fd, 'ReAct')
    assert hasattr(fd, 'register')
    assert hasattr(fd, 'funnier')


def test_version():
    """Test that version is available."""
    assert hasattr(fd, '__version__')
    assert isinstance(fd.__version__, str)


@dataclass
class StatsResult:
    """Test dataclass for structured output."""
    mean_value: float
    count: int


def test_signature_creation():
    """Test that signatures are created correctly."""
    
    @fd.funky
    def test_func(numbers: List[float], threshold: float) -> StatsResult:
        """Test function for signature creation."""
        return StatsResult
    
    # Check that the function has a signature
    assert hasattr(test_func, 'signature')
    
    # Check input fields
    input_fields = test_func.signature.input_fields
    assert 'numbers' in input_fields
    assert 'threshold' in input_fields
    
    # Check output fields
    output_fields = test_func.signature.output_fields
    assert 'StatsResult_mean_value' in output_fields
    assert 'StatsResult_count' in output_fields


def test_tuple_return():
    """Test tuple return type handling."""
    
    @fd.funky
    def test_tuple_func(x: int, y: int) -> tuple[int, str]:
        """Test function with tuple return."""
        result = "The sum"
        message = "Success message"
        return result, message
    
    # Check that the function has a signature
    assert hasattr(test_tuple_func, 'signature')
    
    # Check output fields exist
    output_fields = test_tuple_func.signature.output_fields
    assert len(output_fields) == 2


if __name__ == "__main__":
    pytest.main([__file__]) 