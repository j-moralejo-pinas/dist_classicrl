=========
Changelog
=========

Sources to write the changelog:
- https://keepachangelog.com/en/1.0.0/
- https://semver.org/

Version 0.0
===========

- Finishing the initial version of the project.
- Missing:
    - Tests
    - Documentation
    - Examples
    - CI/CD
    - Code quality checks
    - Versioning
    - Packaging
    - Release notes
- Missing Features:
    - Experience replay.

- Bugs:
    - More cores/vectorized envs makes the training unstable.

Version 0.1
===========

- ``Added``: Basic Q-learning algorithm that chooses iterative/vectorized operations optimally.
- ``Added``: 3 runtime modes: single-process, multi-process and distributed (mpi)
- ``Added``: Rigged two-armed bandit environment for testing.
- ``Added``: Random ticktacktoe environment for testing.
- ``Added``: Benchmarking scripts for comparing runtime modes.
- ``Added``: Constant, linear and exponential schedules.
- ``Added``: Dummy vec env wrapper.
- ``Added``: Flattening multi-discrete spaces wrapper.
- ``Added``: Tests for q-learning, runtimes and environments.
- ``Added``: CI/CD with github actions and precommit hooks.
- ``Added``: Documentation with sphinx and hosting with github pages.
