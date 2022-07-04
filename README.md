# <img src="https://raw.githubusercontent.com/streamlet-dev/tributary/main/docs/img/icon.png" width="300">
Python Data Streams

[![Build Status](https://github.com/streamlet-dev/tributary/workflows/Build%20Status/badge.svg?branch=main)](https://github.com/streamlet-dev/tributary/actions?query=workflow%3A%22Build+Status%22)
[![Coverage](https://codecov.io/gh/streamlet-dev/tributary/branch/main/graph/badge.svg)](https://codecov.io/gh/streamlet-dev/tributary)
[![PyPI](https://img.shields.io/pypi/l/tributary.svg)](https://pypi.python.org/pypi/tributary)
[![PyPI](https://img.shields.io/pypi/v/tributary.svg)](https://pypi.python.org/pypi/tributary)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/streamlet-dev/tributary/main?urlpath=lab)


Tributary is a library for constructing dataflow graphs in python. Unlike many other DAG libraries in python ([airflow](https://airflow.apache.org), [luigi](https://luigi.readthedocs.io/en/stable/), [prefect](https://docs.prefect.io), [dagster](https://docs.dagster.io), [dask](https://dask.org), [kedro](https://github.com/quantumblacklabs/kedro), etc), tributary is not designed with data/etl pipelines or scheduling in mind. Instead, tributary is more similar to libraries like [mdf](https://github.com/man-group/mdf), [pyungo](https://github.com/cedricleroy/pyungo), [streamz](https://streamz.readthedocs.io/en/latest/), or [pyfunctional](https://github.com/EntilZha/PyFunctional), in that it is designed to be used as the implementation for a data model. One such example is the [greeks](https://github.com/streamlet-dev/greeks) library, which leverages tributary to build data models for [options pricing](https://www.investopedia.com/articles/optioninvestor/07/options_beat_market.asp). 

![](https://raw.githubusercontent.com/streamlet-dev/tributary/main/docs/img/example.gif)


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
![](https://raw.githubusercontent.com/streamlet-dev/tributary/main/docs/img/streaming/dagred3.gif)

Here green indicates executing, yellow indicates stalled for backpressure, and red indicates that `StreamEnd` has been propogated (e.g. stream has ended).

## Lazy
![](https://raw.githubusercontent.com/streamlet-dev/tributary/main/docs/img/lazy/dagred3.gif)

Here green indicates executing, and red indicates that the node is dirty. Note the the determination if a node is dirty is also done lazily (we can check with `isDirty` whcih will update the node's graph state.

## Catalog
See the [CATALOG](CATALOG.md) for a full list of functions, transforms, sources, and sinks.

## Support / Contributors
Thanks to the following organizations for providing code or financial support.


<a href="https://nemoulous.com"><img src="https://raw.githubusercontent.com/streamlet-dev/tributary/main/docs/img/nem.png" width="50"></a>

<a href="https://nemoulous.com">Nemoulous</a>

## License
This software is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.

## Alternatives
Here is an incomplete list of libraries which implement similar/overlapping functionality

- [man-group/mdf](https://github.com/man-group/mdf)
- [cedricleroy/pyungo](https://github.com/cedricleroy/pyungo)
- [python-streamz/streamz](https://github.com/python-streamz/streamz)
- [EntilZha/pyfunctional](https://github.com/EntilZha/PyFunctional)
- [stitchfix/hamilton](https://github.com/stitchfix/hamilton)

