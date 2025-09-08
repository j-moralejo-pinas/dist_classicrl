#!/usr/bin/env bash
set -euo pipefail

src/dist_classicrl/benchmarks/run_comprehensive_throughput_benchmarks.sh
python src/dist_classicrl/benchmarks/analyze_throughput_benchmarks.py