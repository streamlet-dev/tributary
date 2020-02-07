.. tributary documentation master file, created by
   sphinx-quickstart on Fri Jan 12 22:07:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tributary
=========

Python Streams

|Build Status| |GitHub| |codecov| |BCH compliance| |PyPI| |PyPI|
|Docs|

Python Data Streams
-------------------

|image7|

Stream Types
============

Tributary offers several kinds of streams:

Reactive
--------

These are synchronous, reactive data streams, built using python
generators

Asynchronous
------------

These are asynchronous, reactive data streams, built using asynchronous
python generators

Functional
----------

These are functional streams, built by currying python functions
(callbacks)

Lazy (work in progress)
-----------------------

These are lazily-evaluated python streams, where outputs are propogated
only as inputs change.

Examples
========

.. _reactive-1:

Reactive
--------

Simple Example
~~~~~~~~~~~~~~

|image8|

More Complex Example
~~~~~~~~~~~~~~~~~~~~

|image9|

Rolling Mean
~~~~~~~~~~~~

|image10|

Custom Calculations and Window Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|image11|

Sources
-------

WebSocket
~~~~~~~~~

|image12|

HTTP
~~~~

|image13|

SocketIO
~~~~~~~~

|image14|

Kafka
~~~~~~~~

|image15|

.. |Build Status| image:: https://travis-ci.org/timkpaine/tributary.svg?branch=master
   :target: https://travis-ci.org/timkpaine/tributary
.. |GitHub| image:: https://img.shields.io/github/issues/timkpaine/tributary.svg
   :target: https://github.com/timkpaine/tributary/issues
.. |codecov| image:: https://codecov.io/gh/timkpaine/tributary/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/timkpaine/tributary
.. |BCH compliance| image:: https://bettercodehub.com/edge/badge/timkpaine/tributary?branch=master
   :target: https://bettercodehub.com/
.. |PyPI| image:: https://img.shields.io/pypi/l/tributary.svg
   :target: https://pypi.python.org/pypi/tributary
.. |PyPI| image:: https://img.shields.io/pypi/v/tributary.svg
   :target: https://pypi.python.org/pypi/tributary
.. |Docs| image:: https://img.shields.io/readthedocs/tributary.svg
   :target: https://tributary.readthedocs.io
.. |image7| image:: img/example.gif
.. |image8| image:: img/reactive/example1.png
.. |image9| image:: img/reactive/example2.png
.. |image10| image:: img/reactive/example3.png
.. |image11| image:: img/reactive/example4.png
.. |image12| image:: img/reactive/ws.png
.. |image13| image:: img/reactive/http.png
.. |image14| image:: img/reactive/sio.png
.. |image15| image:: img/reactive/kafka.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api