Installation Guide
==================

This guide provides step-by-step instructions for installing and setting up the dist_classicrl project template. Choose the installation section that best fits your needs.

.. contents:: Table of Contents
    :local:
    :depth: 2

Prerequisites
-------------

Before installing the project, ensure you have the following requirements:

* **Python 3.13+** (required for this project)
* **Git** for cloning the repository
* **Internet connection** for downloading dependencies

The project has been tested on Linux, macOS, and Windows systems.

User Installation
=================

This section is for users who want to use the project template without modifying the source code.

Quick Start
-----------

1. **Clone the Repository**

    Clone the project repository from GitHub::

        git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
        cd dist_classicrl

2. **Set Up Virtual Environment (Recommended)**

    While not mandatory, using a virtual environment is highly recommended to avoid dependency conflicts::

        # Using conda (recommended)
        conda create -n env_config python=3.x
        conda activate env_config

        # OR using venv
        python -m venv venv
        # On Linux/macOS:
        source venv/bin/activate
        # On Windows:
        venv\Scripts\activate

3. **Install the Package**

    Install the project and its dependencies::

        pip install -e .

4. **Verify Installation**

    Test that the installation was successful::

        python -c "import dist_classicrl; print('Installation successful!')"

5. **Install MPI (Optional)**

    If you plan to use distributed training, install MPI::

        # On Linux
        sudo apt-get install libopenmpi-dev openmpi-bin

        # On macOS
        brew install open-mpi

        # On Windows
        Download and install Microsoft MPI from their official site.

    Finally, verify the installation::

        python -c "import mpi4py; print('MPI4Py installation successful!')"

Developer Installation
======================

This section is for developers who want to contribute to the project or modify the source code.

Development Setup
-----------------

1. **Clone and Navigate**

    ::

        git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
        cd dist_classicrl

2. **Set Up Development Environment**

    Create a virtual environment (recommended)::

        conda create -n dist_classicrl-dev python=3.13
        conda activate dist_classicrl-dev

3. **Install MPI**

    If you plan to use distributed training, install MPI::

        # On Linux
        sudo apt-get install libopenmpi-dev openmpi-bin

        # On macOS
        brew install open-mpi

        # On Windows
        Download and install Microsoft MPI from their official site.

4. **Install in Development Mode**

    Install the package with development dependencies::

        pip install -e ".[dev,docs]"

    This installs the project in editable mode with all development tools including:

   * ``pytest`` - Testing framework
   * ``pyright`` - Type checking
   * ``pre-commit`` - Git hooks for code quality
   * ``ruff`` - Fast Python linter and formatter
   * ``pydoclint`` - Documentation linting
   * ``docformatter`` - Documentation formatting
   * ``pytest-cov`` - Test coverage
   * ``pyupgrade`` - Code modernization
   * ``sphinx`` - Documentation generation
   * ``sphinx-autoapi`` - Automatic API documentation generation

5. **Set Up Pre-commit Hooks**

    Install pre-commit hooks to ensure code quality::

        pre-commit install

6. **Configure Type Checking**

    Link your development environment to Pyright for proper type checking. Create a ``pyrightconfig.local.json`` file in the project root::

        {
            "venvPath": "/path/to/your/conda/envs",
            "venv": "dist_classicrl-dev"
        }

    Replace ``/path/to/your/conda/envs`` with your actual conda environments path (e.g., ``/home/username/miniconda3/envs`` or ``/home/username/micromamba/envs``).

7. **Configure Environment**

    Set the ``PYTHONPATH`` environment variable::

        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

    Or add this to your shell profile (``~/.bashrc``, ``~/.zshrc``, etc.).

8. **Verify Installation**

    Test that the development installation was successful::

        python -c "import dist_classicrl; print('Development installation successful!')"
        pytest --version
        ruff --version
        pyright --version

Troubleshooting
===============

**Common Issues**

**Import Errors**

If you encounter import errors, ensure the ``PYTHONPATH`` is set correctly::

    export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

**Virtual Environment Issues**

If you have issues with virtual environments, try::

    # For conda environments
    conda info --envs  # List all environments
    conda activate dist_classicrl-dev  # Activate the environment

    # For venv environments
    which python  # Check which Python you're using
    pip list  # Check installed packages

**Getting Help**

* Check the project's GitHub issues: https://github.com/j-moralejo-pinas/dist_classicrl/issues
* Review the documentation for detailed usage examples
* Ensure all dependencies are correctly installed

Next Steps
==========

After successful installation, you can:

1. **Read the Documentation**: Navigate through the user guides for detailed usage instructions

For detailed usage instructions and examples, refer to the documentation:
* :doc:`index` - Project overview and documentation
* :doc:`contributing` - How to contribute to the project
