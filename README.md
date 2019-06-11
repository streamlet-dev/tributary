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


# Stream Types
Tributary offers several kinds of streams:

## Reactive
These are synchronous, reactive data streams, built using python generators. They are designed to mimic complex event processors in terms of event ordering.

## Asynchronous
These are synchronous, reactive data streams, built using asynchronous python generators. They are a variant of the reactive streams, but should offer performance improvements over the non-asynchronous variants. 

## Functional
These are functional streams, built by currying python functions (callbacks). 

## Lazy
These are lazily-evaluated python streams, where outputs are propogated only as inputs change.

## Event Loop
TODO
These function as tornado based event-loop based streams similar to streamz.

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
