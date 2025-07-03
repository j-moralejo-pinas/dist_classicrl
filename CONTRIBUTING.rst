============
Contributing
============

Welcome to ``dist_classicrl`` contributor's guide! 🎉

This document will help you get started with contributing to the distributed classical
reinforcement learning library. We welcome all types of contributions, from bug reports
and feature requests to code improvements and documentation enhancements.

If you are new to using git_ or have never collaborated on an open source project before,
please check out the excellent `guide created by FreeCodeCamp`_ and `contribution-guide.org`_.

**Code of Conduct**: All contributors are expected to be **open, considerate, reasonable,
and respectful**. When in doubt, the `Python Software Foundation's Code of Conduct`_
provides excellent guidelines.

Quick Start for Contributors
=============================

For those eager to get started quickly:

.. code-block:: bash

    # 1. Fork the repository on GitHub
    git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
    cd dist_classicrl

    # 2. Create a development environment
    pip install -e ".[dev]"
    pre-commit install

    # 3. Create a feature branch
    git checkout -b feature/my-awesome-feature

    # 4. Make your changes and test them
    pytest tests/

    # 5. Submit a pull request
   git push origin feature/my-awesome-feature


Ways to Contribute
==================

🐛 **Bug Reports**
------------------

Found a bug? Please check the `issue tracker`_ first to see if it's already reported.
When reporting a new bug, please include:

* **Operating system and Python version**
* **dist_classicrl version** (``python -c "import dist_classicrl; print(dist_classicrl.__version__)"`)
* **MPI configuration** (if using distributed features)
* **Minimal code example** that reproduces the issue
* **Expected vs. actual behavior**
* **Error messages and stack traces**

Example bug report template::

    **Environment:**
    - OS: Ubuntu 22.04
    - Python: 3.9.15
    - dist_classicrl: 0.1.0
    - MPI: OpenMPI 4.1.4

    **Bug Description:**
    ParallelQLearning crashes when using more than 4 environments

    **Minimal Example:**
    ```python
    # Code that reproduces the issue
    ```

    **Expected:** Should train successfully
    **Actual:** Crashes with multiprocessing error

📚 **Documentation Improvements**
---------------------------------

Help us improve the documentation! You can:

* Fix typos and grammar issues
* Add missing docstrings to functions and classes
* Improve code examples
* Add tutorials for advanced use cases
* Translate documentation

Documentation uses Sphinx_ and can be built locally with::

    pip install -e ".[docs]"
    cd docs
    make html
    # Open docs/_build/html/index.html in your browser

💡 **Feature Requests**
-----------------------

Have an idea for a new feature? Great! Please:

1. Check existing issues to avoid duplicates
2. Describe the problem your feature would solve
3. Explain your proposed solution
4. Consider implementation complexity and maintenance burden

🔧 **Code Contributions**
-------------------------

We welcome code contributions! Areas where help is especially appreciated:

* **New algorithms**: SARSA, Expected SARSA
* **Performance optimizations**: Vectorization, memory efficiency
* **Distributed training**: Improved MPI coordination, fault tolerance
* **Environment integrations**: PettingZoo support
* **Testing**: Edge cases, integration tests, performance benchmarks

Development Setup
=================

Environment Setup
-----------------

1. **Fork and Clone**

   Fork the repository on GitHub, then clone your fork:

   .. code-block:: bash

       git clone https://github.com/j-moralejo-pinas/dist_classicrl.git
       cd dist_classicrl

2. **Create Virtual Environment**

   We recommend using a virtual environment to avoid dependency conflicts:

   .. code-block:: bash

       # Using venv
       python -m venv dist_classicrl_env
       source dist_classicrl_env/bin/activate  # On Windows: dist_classicrl_env\Scripts\activate

       # OR using conda
       conda create -n dist_classicrl python=3.13
       conda activate dist_classicrl

3. **Install Development Dependencies**

   .. code-block:: bash

       pip install -e ".[dev]"

   This installs the package in editable mode with all development dependencies.

4. **Set Up Pre-commit Hooks**

   .. code-block:: bash

       pre-commit install

   This automatically runs code reformatting. Code quality checks, and linting
   are not enforced on commit, but they are in the CI pipeline. If you want to run them manually,
   you can uncomment the `ruff`, `doclint`, and `pyright` hooks in `.pre-commit-config.yaml`,
   and run:
   ... code-block:: bash
       pre-commit run --all-files

MPI Development (Optional)
--------------------------

For working on distributed training features, install any MPI implementation
and the `mpi4py` package:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install libopenmpi-dev
    pip install mpi4py

    # macOS
    brew install open-mpi
    pip install mpi4py

    # Test MPI installation
    mpirun -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"

Project Architecture
====================

Understanding the codebase structure will help you contribute effectively:

**Core Components:**

* ``src/dist_classicrl/algorithms/``

  * ``base_algorithms/``: Core Q-Learning implementations with different optimizations
  * ``runtime/``: Execution strategies (single-thread, parallel, distributed)
  * ``buffers/``: Experience replay (future expansion)

* ``src/dist_classicrl/environments/``

  * ``custom_env.py``: Abstract base for custom environments with one Q-table multi-agent support
  * ``tiktaktoe_mod.py``: Example environment for testing

* ``src/dist_classicrl/wrappers/``: Environment adapters and transformations
* ``src/dist_classicrl/utils.py``: Utility functions for action space handling

**Design Principles:**

1. **Modularity**: Each algorithm and execution mode is self-contained
2. **Performance**: Vectorized operations preferred over loops where possible
3. **Scalability**: Support for single-thread to distributed execution
4. **Standards Compliance**: Compatible with Gymnasium and PettingZoo
5. **Extensibility**: Easy to add new algorithms and environments

Development Workflow
====================

1. **Create a Feature Branch**

   .. code-block:: bash

       git checkout -b feature/descriptive-name
       # or
       git checkout -b bugfix/issue-123

2. **Make Your Changes**

   * Write clear, documented code
   * Follow existing code style and patterns
   * Add type hints where appropriate
   * Update docstrings for public APIs

3. **Write Tests**

   * Add unit tests for new functionality
   * Update existing tests if needed
   * Ensure good test coverage
   * Test both single-threaded and parallel modes when applicable

   .. code-block:: bash

       # Run tests locally
       pytest tests/

       # Run specific test categories
       pytest tests/dist_classicrl/algorithms/
       pytest tests/dist_classicrl/environments/

       # Run with coverage
       pytest --cov=dist_classicrl tests/

4. **Test MPI Features (if applicable)**

   .. code-block:: bash

       # Run MPI tests
       mpirun -n 3 python -m pytest tests/dist_classicrl/algorithms/runtime/test_q_learning_async_dist.py::TestDistAsyncQLearningMPI

       # Or use the test runner
       bash tests/dist_classicrl/algorithms/runtime/run_runtime_tests.sh

5. **Run Performance Benchmarks**

   If you've modified core algorithms, run performance tests:

   .. code-block:: bash

       cd dev_tests
       python perf_test.py

6. **Check Code Quality**

   .. code-block:: bash

       # Pre-commit will run automatically, but you can run manually:
       pre-commit run --all-files

       # Or run individual tools:
       ruff check src/ tests/
       ruff format src/ tests/
       pyright src/

7. **Update Documentation**

   * Update docstrings for new functions/classes
   * Add examples to the main documentation if needed
   * Update README.rst if adding major features

   .. code-block:: bash

       # Build docs locally
       cd docs
       make html
       # Open docs/_build/html/index.html

8. **Commit Your Changes**

   Write clear, descriptive commit messages:

   .. code-block:: bash

       git add .
       git commit -m "feat: add SARSA algorithm implementation

       - Implement SARSA with epsilon-greedy policy
       - Add comprehensive unit tests
       - Update documentation with usage examples
       - Benchmark performance vs Q-Learning"

9. **Push and Create Pull Request**

   .. code-block:: bash

       git push origin feature/descriptive-name

   Then create a pull request on GitHub to dev with:

   * Clear description of changes
   * Link to relevant issues
   * Screenshots/examples if applicable
   * Mention any breaking changes

**Code Review Process:**

* All changes require review by at least one maintainer
* Focus on code quality, performance impact, and maintainability
* Ensure comprehensive test coverage
* Verify documentation is updated

**Issue Triage:**

* Label issues appropriately (bug, enhancement, documentation, etc.)
* Assign priority levels (critical, high, medium, low)
* Link related issues and pull requests
* Close stale issues after warning period

Links and References
====================

.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/j-moralejo-pinas/dist_classicrl
.. _issue tracker: https://github.com/j-moralejo-pinas/dist_classicrl/issues

.. |virtualenv| replace:: ``virtualenv``
.. |pre-commit| replace:: ``pre-commit``
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
