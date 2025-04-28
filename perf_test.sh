#!/bin/bash

# Performance testing script for CUDA Flocking
# Tests multiple boid counts across different implementations

EXECUTABLE="./build/bin/cis565_boids"

# Array of boid counts to test
BOID_COUNTS=("10000" "50000" "100000" "500000" "1000000" "5000000")

# Function to run each test
run_test() {
    echo "========================================"
    echo "Running test with $1 boids"
    echo "Implementation: $2"
    echo "========================================"
    timeout 30s $EXECUTABLE $1 -p 2>&1 | tee -a results_$2.txt
    echo "" >> results_$2.txt
    echo "Test completed for $1 boids using $2 implementation."
    echo ""
    # Allow GPU to cool down between tests
    sleep 2
}

# Make sure we have a fresh build
cd "$(dirname "$0")"
mkdir -p build
cd build
cmake .. && make

# Clear any previous results
cd ..
rm -f results_*.txt

# Create result headers
echo "COHERENT GRID IMPLEMENTATION RESULTS" > results_coherent.txt
echo "===================================" >> results_coherent.txt
echo "" >> results_coherent.txt

echo "SCATTERED GRID IMPLEMENTATION RESULTS" > results_scattered.txt
echo "====================================" >> results_scattered.txt
echo "" >> results_scattered.txt

echo "NAIVE IMPLEMENTATION RESULTS" > results_naive.txt
echo "=========================" >> results_naive.txt
echo "" >> results_naive.txt

# Run tests for Coherent Grid implementation
for count in "${BOID_COUNTS[@]}"; do
    # Coherent Grid implementation (default in our build)
    run_test $count "coherent"
done

# Recompile for Scattered Grid implementation
cd build
cmake -DUNIFORM_GRID=1 -DCOHERENT_GRID=0 .. && make

# Run tests for Scattered Grid implementation
cd ..
for count in "${BOID_COUNTS[@]}"; do
    run_test $count "scattered"
done

# Only run naive implementation for smaller boid counts as it's very slow for large counts
cd build
cmake -DUNIFORM_GRID=0 -DCOHERENT_GRID=0 .. && make

cd ..
for count in "${BOID_COUNTS[@]:0:3}"; do  # Only use first 3 counts for naive
    run_test $count "naive"
done

echo "All performance tests completed!"
echo "Results saved in results_*.txt files"

# Restore default build
cd build
cmake .. && make
