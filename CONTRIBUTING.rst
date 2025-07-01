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
    git clone https://github.com/YOUR-USERNAME/dist_classicrl.git
    cd dist_classicrl

    # 2. Create a development environment
    pip install -e ".[dev]"
    pre-commit install

    # 3. Create a feature branch
    git checkout -b feature/my-awesome-feature

    # 4. Make your changes and test them
    pytest tests/

    # 5. Submit a pull request

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

* **New algorithms**: SARSA, Expected SARSA, Deep Q-Learning
* **Performance optimizations**: Vectorization, memory efficiency
* **Distributed training**: Improved MPI coordination, fault tolerance
* **Environment integrations**: More Gymnasium/PettingZoo environments
* **Testing**: Edge cases, integration tests, performance benchmarks

Development Setup
=================

Environment Setup
-----------------

1. **Fork and Clone**

   Fork the repository on GitHub, then clone your fork:

   .. code-block:: bash

       git clone https://github.com/YOUR-USERNAME/dist_classicrl.git
       cd dist_classicrl

2. **Create Virtual Environment**

   We recommend using a virtual environment to avoid dependency conflicts:

   .. code-block:: bash

       # Using venv
       python -m venv dist_classicrl_env
       source dist_classicrl_env/bin/activate  # On Windows: dist_classicrl_env\Scripts\activate

       # OR using conda
       conda create -n dist_classicrl python=3.9
       conda activate dist_classicrl

3. **Install Development Dependencies**

   .. code-block:: bash

       pip install -e ".[dev]"

   This installs the package in editable mode with all development dependencies:

   * ``pytest`` and ``pytest-cov`` for testing
   * ``pre-commit`` for code quality hooks
   * ``ruff`` for linting and formatting
   * ``pyright`` for type checking
   * ``sphinx`` for documentation

4. **Set Up Pre-commit Hooks**

   .. code-block:: bash

       pre-commit install

   This automatically runs code quality checks before each commit, including:

   * Code formatting with ``ruff``
   * Import sorting
   * Type checking with ``pyright``
   * Documentation linting

MPI Development (Optional)
--------------------------

For working on distributed training features, install MPI:

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

  * ``custom_env.py``: Abstract base for custom environments
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

   Then create a pull request on GitHub with:

   * Clear description of changes
   * Link to relevant issues
   * Screenshots/examples if applicable
   * Mention any breaking changes
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/dist_classicrl.git
    cd dist_classicrl

#. You should run::

    pip install -U pip setuptools -e .

   to be able to import the package under development in the Python REPL.

   .. todo:: if you are not using pre-commit, please remove the following item:

#. Install |pre-commit|_::

    pip install pre-commit
    pre-commit install

   ``dist_classicrl`` comes with a lot of hooks configured to automatically help the
   developer to check the code being written.

Implement your changes
----------------------

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the main branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.

   .. todo:: if you are not using pre-commit, please remove the following item:

   Please make sure to see the validation messages from |pre-commit|_ and fix
   any eventual issues.
   This should automatically use flake8_/black_ to check/fix the code style
   in a way that is compatible with the project.

   .. important:: Don't forget to add unit tests and documentation in case your
      contribution adds an additional feature and is not just a bugfix.

      Moreover, writing a `descriptive commit message`_ is highly recommended.
      In case of doubt, you can check the commit history with::

         git log --graph --decorate --pretty=oneline --abbrev-commit --all

      to look for recurring communication patterns.

#. Please check that your changes don't break any unit tests with::

    tox

   (after having installed |tox|_ with ``pip install tox`` or ``pipx``).

   You can also use |tox|_ to run several other pre-configured tasks in the
   repository. Try ``tox -av`` to see a list of the available checks.

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.

   .. todo:: if you are using GitHub, you can uncomment the following paragraph

      Find more detailed information in `creating a PR`_. You might also want to open
      the PR as a draft first and mark it as ready for review after the feedbacks
      from the continuous integration (CI) system or any required fixes.


Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.

#. Sometimes |tox|_ misses out when new dependencies are added, especially to
   ``setup.cfg`` and ``docs/requirements.txt``. If you find any problems with
   missing dependencies when running a command with |tox|_, try to recreate the
   ``tox`` environment using the ``-r`` flag. For example, instead of::

    tox -e docs

   Try running::

    tox -r -e docs

#. Make sure to have a reliable |tox|_ installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run::

    tox --version
    # OR
    which tox

   If you have trouble and are seeing weird errors upon running |tox|_, you can
   also try to create a dedicated `virtual environment`_ with a |tox|_ binary
   freshly installed. For example::

    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tox
    .venv/bin/tox -e all

#. `Pytest can drop you`_ in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``tox -- -k <NAME OF THE FALLING TEST> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


**Release Process:**

For maintainers with PyPI access, follow these steps to release a new version:

1. **Prepare Release**

   .. code-block:: bash

       # Ensure all tests pass
       pytest tests/
       bash tests/dist_classicrl/algorithms/runtime/run_runtime_tests.sh

       # Update version and changelog
       # Version is managed by setuptools_scm automatically

2. **Create Release Tag**

   .. code-block:: bash

       git tag v1.2.3
       git push origin v1.2.3

3. **Build and Upload**

   .. code-block:: bash

       # Clean previous builds
       rm -rf dist/ build/

       # Build distribution
       python -m build

       # Upload to PyPI (requires API token)
       python -m twine upload dist/*

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
.. |tox| replace:: ``tox``

.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
