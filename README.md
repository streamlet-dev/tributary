# <a href="https://tributary.readthedocs.io"><img src="docs/img/icon.png" width="300"></a>
Python Data Streams

[![Build Status](https://dev.azure.com/tpaine154/tributary/_apis/build/status/timkpaine.tributary?branchName=master)](https://dev.azure.com/tpaine154/tributary/_build/latest?definitionId=2&branchName=master)
[![GitHub issues](https://img.shields.io/github/issues/timkpaine/tributary.svg)](https://github.com/timkpaine/tributary/issues)
[![Coverage](https://img.shields.io/azure-devops/coverage/tpaine154/tributary/2)](https://dev.azure.com/tpaine154/tributary/_build?definitionId=2&_a=summary)
[![BCH compliance](https://bettercodehub.com/edge/badge/timkpaine/tributary?branch=master)](https://bettercodehub.com/)
[![PyPI](https://img.shields.io/pypi/l/tributary.svg)](https://pypi.python.org/pypi/tributary)
[![PyPI](https://img.shields.io/pypi/v/tributary.svg)](https://pypi.python.org/pypi/tributary)
[![Docs](https://img.shields.io/readthedocs/tributary.svg)](https://tributary.readthedocs.io)

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/example.gif)


# Installation
Install from pip:

`pip install tributary`

or from source

`python setup.py install`

# Stream Types
Tributary offers several kinds of streams:

## Streaming
These are synchronous, reactive data streams, built using asynchronous python generators. They are designed to mimic complex event processors in terms of event ordering.

## Functional
These are functional streams, built by currying python functions (callbacks). 

## Lazy
These are lazily-evaluated python streams, where outputs are propogated only as inputs change.

# Examples
- [Streaming](docs/examples/streaming.md)
- [Lazy](docs/examples/lazy.md)

# Sources and Sinks
## Sources
- python function/generator/async function/async generator
- random
- file
- kafka
- websocket
- http
- socket io

## Sinks
- file
- kafka
- http
- websocket
- TODO socket io

# Transforms
- Delay - Streaming wrapper to delay a stream
- Apply - Streaming wrapper to apply a function to an input stream
- Window - Streaming wrapper to collect a window of values
- Unroll - Streaming wrapper to unroll an iterable stream
- UnrollDataFrame - Streaming wrapper to unroll a dataframe into a stream
- Merge - Streaming wrapper to merge 2 inputs into a single output
- ListMerge - Streaming wrapper to merge 2 input lists into a single output list
- DictMerge - Streaming wrapper to merge 2 input dicts into a single output dict. Preference is given to the second input (e.g. if keys overlap)
- Reduce - Streaming wrapper to merge any number of inputs

# Calculations
- Noop
- Negate
- Invert
- Add
- Sub
- Mult
- Div
- RDiv
- Mod
- Pow
- Sum
- Average
- Not
- And
- Or
- Equal
- NotEqual
- Less
- LessOrEqual
- Greater
- GreaterOrEqual
- Log
- Sin
- Cos
- Tan
- Arcsin
- Arccos
- Arctan
- Sqrt
- Abs
- Exp
- Erf
- Int
- Float
- Bool
- Str
- Len

# Rolling
- RollingCount - Node to count inputs
- RollingMin - Node to take rolling min of inputs
- RollingMax - Node to take rolling max of inputs
- RollingSum - Node to take rolling sum inputs
- RollingAverage - Node to take the running average
