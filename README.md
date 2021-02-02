# <a href="https://tributary.readthedocs.io"><img src="https://raw.githubusercontent.com/timkpaine/tributary/main/docs/img/icon.png" width="300"></a>
Python Data Streams

[![Build Status](https://github.com/timkpaine/tributary/workflows/Build%20Status/badge.svg?branch=main)](https://github.com/timkpaine/tributary/actions?query=workflow%3A%22Build+Status%22)
[![Coverage](https://codecov.io/gh/timkpaine/tributary/branch/main/graph/badge.svg)](https://codecov.io/gh/timkpaine/tributary)
[![PyPI](https://img.shields.io/pypi/l/tributary.svg)](https://pypi.python.org/pypi/tributary)
[![PyPI](https://img.shields.io/pypi/v/tributary.svg)](https://pypi.python.org/pypi/tributary)
[![Docs](https://img.shields.io/readthedocs/tributary.svg)](https://tributary.readthedocs.io)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/timkpaine/tributary/main?urlpath=lab)


Tributary is a library for constructing dataflow graphs in python. Unlike many other DAG libraries in python ([airflow](https://airflow.apache.org), [luigi](https://luigi.readthedocs.io/en/stable/), [prefect](https://docs.prefect.io), [dagster](https://docs.dagster.io), [dask](https://dask.org), [kedro](https://github.com/quantumblacklabs/kedro), etc), tributary is not designed with data/etl pipelines or scheduling in mind. Instead, tributary is more similar to libraries like [mdf](https://github.com/man-group/mdf), [pyungo](https://github.com/cedricleroy/pyungo), [streamz](https://streamz.readthedocs.io/en/latest/), or [pyfunctional](https://github.com/EntilZha/PyFunctional), in that it is designed to be used as the implementation for a data model. One such example is the [greeks](https://github.com/timkpaine/greeks) library, which leverages tributary to build data models for [options pricing](https://www.investopedia.com/articles/optioninvestor/07/options_beat_market.asp). 

![](https://raw.githubusercontent.com/timkpaine/tributary/main/docs/img/example.gif)


# Installation
Install with pip:

`pip install tributary`

or with conda:

`conda install -c conda-forge tributary`

or from source:

`python setup.py install`

Note: If installing from source or with pip, you'll also need [Graphviz itself](https://www.graphviz.org/download/) if you want to visualize the graph using the `.graphviz()` method.

# Stream Types
Tributary offers several kinds of streams:

## Streaming
These are synchronous, reactive data streams, built using asynchronous python generators. They are designed to mimic complex event processors in terms of event ordering.

## Functional
These are functional streams, built by currying python functions (callbacks). 

## Lazy
These are lazily-evaluated python streams, where outputs are propogated only as inputs change. They are implemented as directed acyclic graphs.

# Examples
- [Streaming](docs/examples/streaming/streaming.md): In this example, we construct a variety of forward propogating reactive graphs.
- [Lazy](docs/examples/lazy/lazy.md): In this example, we construct a variety of lazily-evaluated directed acyclic computation graphs. 
- [Automatic Differentiation](docs/examples/autodiff/autodiff.md): In this example, we use `tributary` to perform automatic differentiation on both lazy and streaming graphs.

# Graph Visualization
You can visualize the graph with Graphviz. All streaming and lazy nodes support a `graphviz` method.

Streaming and lazy nodes also support [ipydagred3](https://github.com/timkpaine/ipydagred3) for live update monitoring.

## Streaming
![](https://raw.githubusercontent.com/timkpaine/tributary/main/docs/img/streaming/dagred3.gif)

Here green indicates executing, yellow indicates stalled for backpressure, and red indicates that `StreamEnd` has been propogated (e.g. stream has ended).

## Lazy
![](https://raw.githubusercontent.com/timkpaine/tributary/main/docs/img/lazy/dagred3.gif)

Here green indicates executing, and red indicates that the node is dirty. Note the the determination if a node is dirty is also done lazily (we can check with `isDirty` whcih will update the node's graph state.

# Sources and Sinks
## Sources
- Python Function/Generator/Async Function/Async Generator
- Curve - yield through an iterable
- Const - yield a constant
- Timer - yield on an interval
- Random - generates a random dictionary of values
- File - streams data from a file, optionally loading each line as a json
- HTTP - polls a url with GET requests, streams data out
- HTTPServer - runs an http server and streams data sent by clients
- Websocket - strams data from a websocket
- WebsocketServer - runs a websocket server and streams data sent by clients
- SocketIO - streams data from a socketIO connection
- SocketIOServer - streams data from a socketIO connection
- SSE - streams data from an SSE connection
- Kafka - streams data from kafka
- Postgres - streams data from postgres

## Sinks
- Foo - data to a python function
- File - data to a file
- HTTP - POSTs data to an url
- HTTPServer - runs an http server and streams data to connections
- Websocket - streams data to a websocket
- WebsocketServer - runs a websocket server and streams data to connections
- SocketIO - streams data to a socketIO connection
- SocketIOServer - runs a socketio server and streams data to connections
- SSE - runs an SSE server and streams data to connections
- Kafka - streams data to kafka
- Postgres - streams data to postgres
- Email - streams data and sends it in emails
- TextMessage - streams data and sends it via text message

# Transforms
## Modulate
- Delay - Streaming wrapper to delay a stream
- Throttle - Streaming wrapper to only tick at most every interval
- Debounce - Streaming wrapper to only tick on new values
- Apply - Streaming wrapper to apply a function to an input stream
- Window - Streaming wrapper to collect a window of values
- Unroll - Streaming wrapper to unroll an iterable stream
- UnrollDataFrame - Streaming wrapper to unroll a dataframe into a stream
- Merge - Streaming wrapper to merge 2 inputs into a single output
- ListMerge - Streaming wrapper to merge 2 input lists into a single output list
- DictMerge - Streaming wrapper to merge 2 input dicts into a single output dict. Preference is given to the second input (e.g. if keys overlap)
- Reduce - Streaming wrapper to merge any number of inputs
- FixedMap - Map input stream to fixed number of outputs
- Subprocess - Open a subprocess and yield results as they come. Can also stream data to subprocess (either instantaneous or long-running subprocess)


## Calculations
Note that `tributary` can also be configured to operate on **dual numbers** for things like lazy or streaming autodifferentiation.

### Arithmetic Operators
- Noop (unary) - Pass input to output
- Negate (unary) - -1 * input
- Invert (unary) - 1/input
- Add (binary) - add 2 inputs
- Sub (binary) - subtract second input from first
- Mult (binary) - multiple inputs
- Div (binary) - divide first input by second
- RDiv (binary) - divide second input by first
- Mod (binary) - first input % second input
- Pow (binary) - first input^second input
- Sum (n-ary) - sum all inputs
- Average (n-ary) - average of all inputs
- Round (unary)
- Floor (unary)
- Ceil (unary)

### Boolean Operators
- Not (unary) - `Not` input
- And (binary) - `And` inputs
- Or (binary) - `Or` inputs

### Comparators
- Equal (binary) - inputs are equal
- NotEqual (binary) - inputs are not equal
- Less (binary) - first input is less than second input
- LessOrEqual (binary) - first input is less than or equal to second input
- Greater (binary) - first input is greater than second input
- GreaterOrEqual (binary) - first input is greater than or equal to second input

### Math
- Log (unary)
- Sin (unary)
- Cos (unary)
- Tan (unary)
- Arcsin (unary)
- Arccos (unary)
- Arctan (unary)
- Sqrt (unary)
- Abs (unary)
- Exp (unary)
- Erf (unary)

### Financial Calculations
- RSI - Relative Strength Index
- MACD - Moving Average Convergence Divergence

## Converters
- Int (unary)
- Float (unary)
- Bool (unary)
- Str (unary)

## Basket Functions
- Len (unary)
- Count (unary)
- Min (unary)
- Max (unary)
- Sum (unary)
- Average (unary)

## Rolling
- RollingCount - Node to count inputs
- RollingMin - Node to take rolling min of inputs
- RollingMax - Node to take rolling max of inputs
- RollingSum - Node to take rolling sum inputs
- RollingAverage - Node to take the running average
- SMA - Node to take the simple moving average over a window
- EMA - Node to take an exponential moving average over a window

## Node Type Converters
- Lazy->Streaming
