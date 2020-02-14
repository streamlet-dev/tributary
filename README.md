# <a href="https://tributary.readthedocs.io"><img src="docs/img/icon.png" width="300"></a>
Python Data Streams

[![Build Status](https://travis-ci.org/timkpaine/tributary.svg?branch=master)](https://travis-ci.org/timkpaine/tributary)
[![GitHub issues](https://img.shields.io/github/issues/timkpaine/tributary.svg)]()
[![codecov](https://codecov.io/gh/timkpaine/tributary/branch/master/graph/badge.svg)](https://codecov.io/gh/timkpaine/tributary)
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

## Reactive
These are synchronous, reactive data streams, built using asynchronous python generators. They are designed to mimic complex event processors in terms of event ordering.

## Functional
These are functional streams, built by currying python functions (callbacks). 

## Event Loop
TODO
These function as tornado based event-loop based streams similar to streamz.

## Lazy
These are lazily-evaluated python streams, where outputs are propogated only as inputs change.

# Examples
- [Reactive](docs/examples/reactive.md)
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
