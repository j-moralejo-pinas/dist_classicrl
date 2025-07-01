============
Installation
============

Requirements
============

**System Requirements:**

- Python 3.8 or higher
- Operating System: Linux, macOS, or Windows
- Memory: At least 4GB RAM (more for large-scale training)
- CPU: Multi-core recommended for parallel training

**Core Dependencies:**

- NumPy >= 1.20.0
- Gymnasium >= 1.0.0

**Optional Dependencies:**

- ``mpi4py`` - For distributed training with MPI
- ``pytest`` - For running tests
- ``sphinx`` - For building documentation

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

This includes testing, documentation, and code quality tools.

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

Common Issues
=============

Import Errors
-------------

If you encounter import errors:

.. code-block:: bash

    # Reinstall with no cache
    pip uninstall dist_classicrl
    pip install --no-cache-dir dist_classicrl

MPI Issues
----------

**"mpi4py not found":**

Ensure MPI is properly installed before installing mpi4py:

.. code-block:: bash

    # Check MPI installation
    which mpirun
    mpirun --version

    # Reinstall mpi4py
    pip uninstall mpi4py
    pip install mpi4py

**MPI version conflicts:**

If you have multiple MPI implementations:

.. code-block:: bash

    # Check which MPI implementation is being used
    python -c "from mpi4py import MPI; print(MPI.Get_library_version())"

**Permission errors on Linux:**

Some systems require additional setup for MPI:

.. code-block:: bash

    # Add to ~/.bashrc or ~/.profile
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

Performance Issues
------------------

**Slow imports:**

If imports are slow, check for network-mounted filesystems or antivirus software.

**Memory errors with large Q-tables:**

Consider using sparse representations or reducing state space size:

.. code-block:: python

    # Use numpy with appropriate dtype
    import numpy as np

    # For smaller Q-tables, use float32 instead of float64
    agent = SingleThreadQLearning(
        state_size=1000,
        action_size=10,
        q_table_dtype=np.float32  # Reduces memory usage
    )

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

    conda create -n dist_classicrl python=3.9
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
