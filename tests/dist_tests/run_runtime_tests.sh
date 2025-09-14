#!/usr/bin/env bash
set -euo pipefail

# Run the distributed pytest suite using MPI with 2 and 3 ranks.
# Usage:
#   bash run_runtime_tests.sh [additional pytest args]

[[ "${DEBUG:-}" == "1" ]] && set -x

# Resolve repo root (prefer git, fallback to relative path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
	:
else
	ROOT_DIR="$(realpath "$SCRIPT_DIR/../../../..")"
fi

# Locate MPI launcher
if command -v mpiexec >/dev/null 2>&1; then
	MPI_RUN=mpiexec
elif command -v mpirun >/dev/null 2>&1; then
	MPI_RUN=mpirun
else
	echo "Error: mpiexec/mpirun not found in PATH" >&2
	exit 127
fi

# Recommended to avoid CPU oversubscription in CI/containers
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

TEST_FILE="$ROOT_DIR/tests/dist_tests/test_q_learning_distributed.py"
PYTEST_BASE_ARGS=("-q" "-rA" "$TEST_FILE")

# Allow passing through extra pytest args
if [[ $# -gt 0 ]]; then
	PYTEST_ARGS=("${PYTEST_BASE_ARGS[@]}" "$@")
else
	PYTEST_ARGS=("${PYTEST_BASE_ARGS[@]}")
fi

echo "[MPI 2 ranks] Running: $MPI_RUN -n 2 python -m pytest ${PYTEST_ARGS[*]}"
$MPI_RUN -n 2 python -m pytest "${PYTEST_ARGS[@]}"
rc2=$?

echo "[MPI 3 ranks] Running: $MPI_RUN -n 3 python -m pytest ${PYTEST_ARGS[*]}"
$MPI_RUN -n 3 python -m pytest "${PYTEST_ARGS[@]}"
rc3=$?

if [[ $rc2 -ne 0 || $rc3 -ne 0 ]]; then
	echo "Distributed tests failed: rc2=$rc2 rc3=$rc3" >&2
	exit 1
fi

echo "Distributed tests passed for 2 and 3 MPI ranks."
