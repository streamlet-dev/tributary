==============
Live
==============

Lantern Live is a framework for live charting inside Jupyter. It utilizes a WebAssembly pivoting engine, `Perspective <https://github.com/jpmorganchase/perspective>`_, along with high performance chart and table libraries `Highcharts <https://www.highcharts.com>`_ and `Hypergrid <https://github.com/fin-hypergrid/core>`_.

Live Tables
==============

.. image:: ./img/live/grid.gif
    :scale: 100%
    :alt: grid.gif

Live Charts
==============

.. image:: ./img/live/line.gif
    :scale: 100%
    :alt: line.gif

Perspective
============
Lantern relies on Perspective to do live pivoting and configuration. 

.. image:: ./img/live/line2.gif
    :scale: 100%
    :alt: line2.gif

Integrations
============
With Lantern and Perspective, it is easy to visualize streaming data. Here, we consume raw price feeds from `IEX <https://iextrading.com>`_ using their websocket library through `pyEX <https://github.com/timkpaine/pyEX>`_. We could easily do some further calculations, and visualize our derived data. 

.. image:: ./img/live/pyEX.gif
    :scale: 100%
    :alt: line2.gif
