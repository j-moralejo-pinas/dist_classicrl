==============
dist_classicrl
==============

.. image:: https://img.shields.io/pypi/v/dist_classicrl.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/dist_classicrl/
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: License
    :target: https://github.com/j-moralejo-pinas/dist_classicrl/blob/main/LICENSE.txt
.. image:: https://img.shields.io/badge/python-3.13+-blue.svg
    :alt: Python Version

**A high-performance Python package for distributed classical reinforcement learning**

dist_classicrl provides scalable implementations of classic reinforcement learning algorithms
with support for single-threaded, parallel (multiprocessing), and distributed (MPI) training.
The library focuses on Q-Learning with optimized vectorized operations and comprehensive
performance benchmarking.

Quick Start
===========

Install the package:

.. code-block:: bash

    pip install dist_classicrl

For detailed examples and tutorials, see :doc:`tutorials`.

Key Features
============

🚀 **Multiple Execution Modes**
    - Single-threaded for development and debugging
    - Parallel multiprocessing for local scaling
    - MPI distributed training for cluster deployment

⚡ **High Performance**
    - Vectorized NumPy operations
    - Optimized algorithm implementations
    - Performance benchmarking and profiling

🎮 **Multi-Agent Support**
    - Built-in support for multi-agent environments
    - Compatible with Gymnasium and PettingZoo (coming soon)

🔧 **Flexible Architecture**
    - Abstract base classes for easy extension
    - Modular design for algorithm composition
    - Custom environment support

Documentation Sections
=======================
========

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    Overview <readme>
    Installation & Quick Start <installation>
    Tutorials <tutorials>
    Performance Benchmarks <benchmarks>

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    Algorithms <user_guide/algorithms>
    Environments <user_guide/environments>
    Performance & Benchmarking <user_guide/performance>
    Distributed Training <user_guide/distributed>

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    API Documentation <autoapi/index>

.. toctree::
    :maxdepth: 2
    :caption: Development

    Contributing <contributing>
    Architecture <development/architecture>
    Testing <development/testing>

.. toctree::
    :maxdepth: 1
    :caption: Project Info

    License <license>
    Authors <authors>
    Changelog <changelog>


Examples
========

For complete examples and tutorials on how to use single-thread, parallel, and distributed training modes, see :doc:`tutorials`.

For performance benchmarking examples, see :doc:`benchmarks`.

Performance Highlights
======================

The library includes extensive performance optimizations:

- **Vectorized Operations**: Up to 10x speedup for large action spaces
- **Memory Efficiency**: Optimized Q-table storage and access patterns
- **Parallel Scaling**: Near-linear speedup with multiple CPU cores
- **Distributed Scaling**: Efficient MPI communication patterns for large-scale training

Algorithm Implementations
=========================

**Q-Learning Variants:**

- **Optimal Q-Learning**: Base implementation with multiple optimization strategies
- **List-based Q-Learning**: List based Q-table for small state spaces
- **NumPy Q-Learning**: Vectorized Q-table using NumPy for large state spaces

**Execution Modes:**

- **Single-threaded**: ``q_learning_single_thread``
- **Parallel**: ``q_learning_parallel`` (multiprocessing)
- **Distributed**: ``q_learning_async_dist`` (MPI)

**Future Algorithms:**

- SARSA and Expected SARSA
- Deep Q-Learning integration
- Multi-agent coordination algorithms

Support and Community
=====================

- **GitHub Issues**: `Report bugs and request features <https://github.com/j-moralejo-pinas/dist_classicrl/issues>`_
- **Documentation**: You're reading it! 📖
- **Contributing**: See :doc:`contributing` for how to get involved

The dist_classicrl project welcomes contributions from the community. Whether you're fixing bugs,
adding features, improving documentation, or sharing your use cases, we'd love to hear from you!


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists
