============
Installation
============

Requirements
============

- ``MPI`` - For distributed training (optional)

To install MPI, follow the instructions below based on your operating system.

Installation Methods
===================

PyPI Installation (Recommended)
--------------------------------

Install the latest stable version from PyPI:

.. code-block:: bash

    pip install dist_classicrl

This installs the core package with all required dependencies.

Development Installation
------------------------

For the latest features or to contribute to development:

.. code-block:: bash

    git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
    cd dist_classicrl
    pip install -e .

For development with all optional dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This includes testing, and code quality tools.

For building documentation locally:

.. code-block:: bash

    pip install -e ".[docs]"



MPI Support (Optional)
======================

For distributed training capabilities, you'll need MPI installed:

Ubuntu/Debian
-------------

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install libopenmpi-dev openmpi-bin
    pip install mpi4py

macOS
-----

Using Homebrew:

.. code-block:: bash

    brew install open-mpi
    pip install mpi4py

Using MacPorts:

.. code-block:: bash

    sudo port install openmpi
    pip install mpi4py

Windows
-------

Windows MPI support requires Microsoft MPI:

1. Download and install `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_
2. Install mpi4py:

.. code-block:: bash

    pip install mpi4py

Conda Installation
------------------

If you prefer conda:

.. code-block:: bash

    conda install -c conda-forge mpi4py
    pip install dist_classicrl

Verification
============

Test your installation:

.. code-block:: python

    import dist_classicrl
    print(f"dist_classicrl version: {dist_classicrl.__version__}")

    # Test basic functionality
    from dist_classicrl.algorithms.runtime.q_learning_single_thread import SingleThreadQLearning
    agent = SingleThreadQLearning(state_size=10, action_size=4)
    print("✓ Single-threaded Q-learning works")

    # Test parallel functionality
    from dist_classicrl.algorithms.runtime.q_learning_parallel import ParallelQLearning
    agent = ParallelQLearning(state_size=10, action_size=4)
    print("✓ Parallel Q-learning works")

Test MPI Installation (if installed):

.. code-block:: bash

    # Test MPI
    mpirun -n 2 python -c "from mpi4py import MPI; print(f'MPI Rank: {MPI.COMM_WORLD.Get_rank()}')"

    # Test distributed Q-learning
    mpirun -n 2 python -c "from dist_classicrl.algorithms.runtime.q_learning_async_dist import DistAsyncQLearning; print('✓ Distributed Q-learning available')"

Virtual Environments
===================

We strongly recommend using virtual environments:

**Using venv:**

.. code-block:: bash

    python -m venv dist_classicrl_env
    source dist_classicrl_env/bin/activate  # On Windows: dist_classicrl_env\Scripts\activate
    pip install dist_classicrl

**Using conda:**

.. code-block:: bash

    conda create -n dist_classicrl python=3.13
    conda activate dist_classicrl
    pip install dist_classicrl

Upgrading
=========

To upgrade to the latest version:

.. code-block:: bash

    pip install --upgrade dist_classicrl

To upgrade to a specific version:

.. code-block:: bash

    pip install dist_classicrl==1.2.3

Uninstalling
============

To completely remove the package:

.. code-block:: bash

    pip uninstall dist_classicrl

If you installed development dependencies:

.. code-block:: bash

    pip uninstall dist_classicrl pytest sphinx ruff pyright pre-commit

Next Steps
==========

After installation, check out:

- :doc:`tutorials` - Step-by-step guides for common use cases
- :doc:`user_guide/algorithms` - Detailed algorithm documentation
- :doc:`autoapi/index` - Complete API reference
- :doc:`user_guide/performance` - Performance optimization tips
