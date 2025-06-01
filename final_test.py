#!/usr/bin/env python3

import funnydspy as fd
import dspy

print('✅ FunnyDSPy version:', fd.__version__)

# Configure DSPy
dspy.configure(lm=dspy.LM('openai/gpt-4.1-nano', cache=False))

@fd.Predict
def test(x: str) -> str:
    return y

# Test funnier
mod = test.module
wrapped = fd.funnier(mod)
result = wrapped('hello')
print('✅ funnier works:', result)

# Test parallelize
parallel_test = fd.parallelize(test)
results = parallel_test([{'x': 'hello'}, {'x': 'world'}])
print('✅ parallelize works:', results)

print('🎉 All functionality working correctly!') 