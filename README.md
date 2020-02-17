# <a href="https://tributary.readthedocs.io"><img src="docs/img/icon.png" width="300"></a>
Python Data Streams

[![Build Status](https://dev.azure.com/tpaine154/tributary/_apis/build/status/timkpaine.tributary?branchName=master)](https://dev.azure.com/tpaine154/tributary/_build/latest?definitionId=2&branchName=master)
[![GitHub issues](https://img.shields.io/github/issues/timkpaine/tributary.svg)]()
[![Coverage](https://img.shields.io/azure-devops/coverage/tpaine154/tributary/2)]()
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

# Math
`(Work in progress)`

## Operations
- unary operators/comparators
- binary operators/comparators

## Rolling
- count
- sum

# Sources and Sinks
`(Work in progress)`

## Sources
- file
- kafka
- websocket
- http
- socket io

## Sinks
- file
- kafka
- http
- TODO websocket
- TODO socket io
