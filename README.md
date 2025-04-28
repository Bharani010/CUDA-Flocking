# CUDA Flocking Simulation

## Overview
This project implements a real-time boid flocking simulation using CUDA to leverage GPU parallelization. The simulation features thousands of autonomous agents (boids) that exhibit emergent flocking behaviors through simple rules of cohesion, separation, and alignment.

## Key Features
- **High-performance CUDA implementation** supporting up to millions of boids
- **Three simulation methods**:
  - Naive (brute force) approach
  - Scattered grid-based neighbor search
  - Coherent grid-based neighbor search with memory optimization
- **Interactive visualization** for real-time observation
- **Performance testing mode** for benchmarking different methods

## Quick Start

### Building the Project
```bash
# Navigate to project directory
cd CUDA-Flocking

# Build the project
make
```

### Running the Visualization
To run the visualization with default settings (5000 boids, naive method):
```bash
./build/bin/cis565_boids
```

To specify simulation method and boid count:
```bash
./build/bin/cis565_boids --mode [naive|scattered|coherent] --boids [count]
```

Example:
```bash
./build/bin/cis565_boids --mode coherent --boids 25000
```

### Controls
- **Left mouse button**: Rotate camera
- **Right mouse button**: Zoom in/out
- **ESC**: Exit the application

## Performance Testing

### Single Method Performance Test
To run performance tests for a specific method and boid count:
```bash
./build/bin/cis565_boids --perf-test [naive|scattered|coherent] [count]
```

Example:
```bash
./build/bin/cis565_boids --perf-test coherent 50000
```

### Full Performance Testing
To test all boid sizes with a specific method:
```bash
./build/bin/cis565_boids --perf-test [naive|scattered|coherent]
```

Example:
```bash
./build/bin/cis565_boids --perf-test coherent
```

Alternatively, use the provided script to run all performance tests:
```bash
./perf_test.sh
```

## Implementation Details

### Optimization Approaches

#### 1. Parameter Optimization
We carefully tuned the flocking parameters to achieve natural-looking behavior while maintaining performance:
- Increased `maxSpeed` from 1.8 to 2.5 for faster and more dynamic movement
- Enhanced alignment factor (`rule3Scale`) from 0.15 to 0.25 for smoother directional changes
- Adjusted cohesion and separation weights for better flocking behavior
- Reduced time step (`dt`) from 0.2 to 0.1 for smoother animation

#### 2. Algorithmic Optimizations

**Naive Implementation:**
- Brute-force approach that compares each boid with every other boid
- O(n²) complexity, suitable for small simulations
- Used squared distances to avoid expensive sqrt operations

**Scattered Grid Implementation:**
- Divides space into a uniform grid for efficient neighbor searching
- Only examines boids in neighboring cells instead of all boids
- Uses Thrust library for efficient sorting of boids by grid cell
- Maintains original data structure with indirect access using indices

**Coherent Grid Implementation:**
- Similar to scattered grid but reorganizes data for memory coherence
- Rearranges boid data to be contiguous in memory based on spatial locality
- Enables more efficient memory access patterns on the GPU
- Swaps pointers instead of copying data back to avoid expensive transfers
- Added explicit synchronization points to ensure data consistency

#### 3. Memory Optimizations
- Used ping-pong buffering for velocity updates to avoid race conditions
- Pre-computed squared distances for performance
- Implemented memory-efficient grid cell indexing
- Optimized memory access patterns for better coalescing

## Technical Details

### Key CUDA Kernels

1. **kernUpdateVelocityBruteForce**: Updates velocities using brute-force approach

2. **kernUpdateVelNeighborSearchScattered**: Updates velocities using grid-based neighbor search with scattered memory layout

3. **kernUpdateVelNeighborSearchCoherent**: Updates velocities using grid-based search with coherent memory layout

4. **kernUpdatePos**: Updates boid positions based on velocities and handles boundary conditions

5. **kernResetAndComputeIndices**: Assigns boids to grid cells and resets grid data structures

6. **kernIdentifyCellStartEnd**: Identifies the start and end indices for each grid cell

7. **kernRearrangeBoidData**: Reorganizes boid data for memory coherence

### Simulation Steps

For the coherent grid implementation (most optimized):

1. Reset grid cell buffers and compute boid grid indices
2. Sort boids by grid cell index
3. Identify start/end indices for each grid cell
4. Rearrange boid data to be coherent in memory
5. Update velocities using neighbor search
6. Update positions using newly calculated velocities
7. Swap buffer pointers for next iteration

## Performance Analysis

The performance hierarchy from least to most efficient:

1. **Naive**: O(n²) complexity, becomes impractical beyond ~10,000 boids
2. **Scattered Grid**: O(n) complexity but with non-coalesced memory access
3. **Coherent Grid**: O(n) complexity with optimized memory access patterns

The coherent grid implementation typically provides 10-100x speedup compared to naive for large boid counts.

Note that optimal performance depends on your specific GPU. For reference, on an NVIDIA GeForce RTX 3080:
- Coherent grid achieves 60+ FPS with 100,000 boids
- Scattered grid achieves 60+ FPS with 50,000 boids
- Naive implementation achieves 60+ FPS with only 5,000 boids


