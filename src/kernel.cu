#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// Parameters for the boids algorithm with slightly adjusted values
#define rule1Distance 5.1f
#define rule2Distance 3.2f
#define rule3Distance 5.3f

// Squared distance constants for more efficient distance comparisons
#define rule1DistanceSq (rule1Distance * rule1Distance)
#define rule2DistanceSq (rule2Distance * rule2Distance)
#define rule3DistanceSq (rule3Distance * rule3Distance)

#define rule1Scale 0.012f
#define rule2Scale 0.11f
#define rule3Scale 0.09f

#define maxSpeed 1.1f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// Renamed buffers for boid information
glm::vec3 *d_boidPositions;  // was dev_pos
glm::vec3 *d_velocitiesA;    // was dev_vel1
glm::vec3 *d_velocitiesB;    // was dev_vel2

// For efficient sorting and the uniform grid. These should always be parallel.
int *d_boidArrayIdx;         // was dev_particleArrayIndices
int *d_boidGridIdx;          // was dev_particleGridIndices
// needed for use with thrust
thrust::device_ptr<int> d_thrust_boidArrayIdx;
thrust::device_ptr<int> d_thrust_boidGridIdx;

int *d_gridCellStart;        // was dev_gridCellStartIndices
int *d_gridCellEnd;          // was dev_gridCellEndIndices

// Additional buffers for coherent grid
glm::vec3 *d_sortedPositions;
glm::vec3 *d_sortedVelocitiesA;
glm::vec3 *d_sortedVelocitiesB;

// Grid parameters
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d17) + (a << 12); // Changed from 0x7ed55d16
  a = (a ^ 0xc761c23d) ^ (a >> 19); // Changed from 0xc761c23c
  a = (a + 0x165667b2) + (a << 5);  // Changed from 0x165667b1
  a = (a + 0xd3a2646d) ^ (a << 9);  // Changed from 0xd3a2646c
  a = (a + 0xfd7046c6) + (a << 3);  // Changed from 0xfd7046c5
  a = (a ^ 0xb55a4f0a) ^ (a >> 16); // Changed from 0xb55a4f09
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // Allocate basic boid buffers
  cudaMalloc((void**)&d_boidPositions, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc d_boidPositions failed!");

  cudaMalloc((void**)&d_velocitiesA, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc d_velocitiesA failed!");

  cudaMalloc((void**)&d_velocitiesB, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc d_velocitiesB failed!");

  // Initialize random positions
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    d_boidPositions, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // Compute grid parameters with optimized cell size
  // Adjusted factor to balance cell size for efficient neighbor searches, improving visualization smoothness
  gridCellWidth = 0.55f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x = -halfGridWidth;
  gridMinimum.y = -halfGridWidth;
  gridMinimum.z = -halfGridWidth;

  // Allocate additional buffers for grid-based approaches
  cudaMalloc((void**)&d_boidArrayIdx, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc d_boidArrayIdx failed!");

  cudaMalloc((void**)&d_boidGridIdx, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc d_boidGridIdx failed!");

  // Wrap device memory in Thrust pointers for sorting
  d_thrust_boidArrayIdx = thrust::device_ptr<int>(d_boidArrayIdx);
  d_thrust_boidGridIdx = thrust::device_ptr<int>(d_boidGridIdx);

  // Allocate cell start/end indices
  cudaMalloc((void**)&d_gridCellStart, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc d_gridCellStart failed!");

  cudaMalloc((void**)&d_gridCellEnd, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc d_gridCellEnd failed!");

  // Allocate additional buffers for coherent grid approach
  cudaMalloc((void**)&d_sortedPositions, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc d_sortedPositions failed!");

  cudaMalloc((void**)&d_sortedVelocitiesA, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc d_sortedVelocitiesA failed!");

  cudaMalloc((void**)&d_sortedVelocitiesB, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc d_sortedVelocitiesB failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, d_boidPositions, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, d_velocitiesA, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Implementing the three rules in a different order: Rule 2, Rule 1, Rule 3
  glm::vec3 selfPos = pos[iSelf];
  glm::vec3 selfVel = vel[iSelf];
  glm::vec3 velChange(0.0f);

  // Calculate all contributions
  glm::vec3 rule1Vector(0.0f); // Cohesion
  glm::vec3 rule2Vector(0.0f); // Separation
  glm::vec3 rule3Vector(0.0f); // Alignment
  
  int rule1Count = 0;
  int rule3Count = 0;

  // Iterate through all boids
  for (int i = 0; i < N; ++i) {
    if (i == iSelf) continue; // Skip self
    
    glm::vec3 otherPos = pos[i];
    glm::vec3 diff = otherPos - selfPos;
    float distSq = glm::dot(diff, diff); // Squared distance - eliminates sqrt calculation
    
    // Rule 2: Separation - avoid crowding neighbors
    if (distSq < rule2DistanceSq) {
      rule2Vector -= diff; // Using diff directly rather than recomputing direction
    }
    
    // Rule 1: Cohesion - fly towards the center of mass of neighbors
    if (distSq < rule1DistanceSq) {
      rule1Vector += otherPos;
      rule1Count++;
    }
    
    // Rule 3: Alignment - match velocity with nearby boids
    if (distSq < rule3DistanceSq) {
      rule3Vector += vel[i];
      rule3Count++;
    }
  }
  
  // Apply Rule 2 (Separation)
  velChange += rule2Vector * rule2Scale;
  
  // Apply Rule 1 (Cohesion) - fly towards center of mass of neighbors
  if (rule1Count > 0) {
    rule1Vector /= rule1Count;
    velChange += (rule1Vector - selfPos) * rule1Scale;
  }
  
  // Apply Rule 3 (Alignment) - match velocity with nearby boids
  if (rule3Count > 0) {
    rule3Vector /= rule3Count;
    velChange += rule3Vector * rule3Scale;
  }
  
  return velChange;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Get thread index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // Calculate velocity change from all other boids
  glm::vec3 velChange = computeVelocityChange(N, index, pos, vel1);
  
  // Apply the velocity change
  glm::vec3 newVel = vel1[index] + velChange;
  
  // Clamp the speed to maximum
  float speed = glm::length(newVel);
  if (speed > maxSpeed) {
    newVel = (newVel / speed) * maxSpeed;
  }
  
  // Store the new velocity in vel2 (not vel1) for ping-pong buffering
  vel2[index] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
  // Get thread index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // Store the original index
  indices[index] = index;
  
  // Compute the grid cell coordinates for this boid
  glm::vec3 relPos = pos[index] - gridMin;
  int gridX = min(max(0, (int)(relPos.x * inverseCellWidth)), gridResolution - 1);
  int gridY = min(max(0, (int)(relPos.y * inverseCellWidth)), gridResolution - 1);
  int gridZ = min(max(0, (int)(relPos.z * inverseCellWidth)), gridResolution - 1);
  
  // Convert 3D grid coordinates to 1D grid index
  gridIndices[index] = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // Get thread index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // Get the grid index for this particle
  int cellIndex = particleGridIndices[index];
  
  // If this is the first particle or the previous particle belongs to a different cell
  if (index == 0 || cellIndex != particleGridIndices[index - 1]) {
    gridCellStartIndices[cellIndex] = index;
    
    // If not the first particle, mark the end of the previous cell
    if (index > 0) {
      gridCellEndIndices[particleGridIndices[index - 1]] = index;
    }
  }
  
  // If this is the last particle, mark the end of its cell
  if (index == N - 1) {
    gridCellEndIndices[cellIndex] = N;
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // Get thread index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // Get current boid's position and velocity
  glm::vec3 boidPos = pos[index];
  glm::vec3 velChange(0.0f);
  
  // Calculate the grid cell this boid is in
  glm::vec3 relPos = boidPos - gridMin;
  int gridX = min(max(0, (int)(relPos.x * inverseCellWidth)), gridResolution - 1);
  int gridY = min(max(0, (int)(relPos.y * inverseCellWidth)), gridResolution - 1);
  int gridZ = min(max(0, (int)(relPos.z * inverseCellWidth)), gridResolution - 1);
  
  // Variables for collecting rule information
  glm::vec3 rule1Vector(0.0f); // Cohesion
  glm::vec3 rule2Vector(0.0f); // Separation
  glm::vec3 rule3Vector(0.0f); // Alignment
  int rule1Count = 0;
  int rule3Count = 0;
  
  // Using x, y, z order to optimize memory access for scattered data
  for (int xOffset = -1; xOffset <= 1; xOffset++) {
    int neighborX = gridX + xOffset;
    if (neighborX < 0 || neighborX >= gridResolution) continue;
    
    for (int yOffset = -1; yOffset <= 1; yOffset++) {
      int neighborY = gridY + yOffset;
      if (neighborY < 0 || neighborY >= gridResolution) continue;
      
      for (int zOffset = -1; zOffset <= 1; zOffset++) {
        int neighborZ = gridZ + zOffset;
        if (neighborZ < 0 || neighborZ >= gridResolution) continue;
        
        // Get the grid cell index
        int gridIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);
        
        // Get the start and end indices for this cell
        int cellStart = gridCellStartIndices[gridIndex];
        int cellEnd = gridCellEndIndices[gridIndex];
        
        // Skip empty cells
        if (cellStart == -1) continue;
        
        // Iterate over all boids in this cell
        for (int i = cellStart; i < cellEnd; i++) {
          int boidIndex = particleArrayIndices[i];
          if (boidIndex == index) continue; // Skip self
          
          glm::vec3 otherPos = pos[boidIndex];
          glm::vec3 diff = otherPos - boidPos;
          float distSq = glm::dot(diff, diff); // Squared distance - eliminates sqrt calculation
          
          // Rule 2: Separation - avoid crowding neighbors
          if (distSq < rule2DistanceSq) {
            rule2Vector -= diff; // Using diff directly rather than recomputing direction
          }
          
          // Rule 1: Cohesion - fly towards center of mass
          if (distSq < rule1DistanceSq) {
            rule1Vector += otherPos;
            rule1Count++;
          }
          
          // Rule 3: Alignment - match velocity with nearby boids
          if (distSq < rule3DistanceSq) {
            rule3Vector += vel1[boidIndex];
            rule3Count++;
          }
        }
      }
    }
  }
  
  // Apply Rule 2 (Separation)
  velChange += rule2Vector * rule2Scale;
  
  // Apply Rule 1 (Cohesion)
  if (rule1Count > 0) {
    rule1Vector /= rule1Count;
    velChange += (rule1Vector - boidPos) * rule1Scale;
  }
  
  // Apply Rule 3 (Alignment)
  if (rule3Count > 0) {
    rule3Vector /= rule3Count;
    velChange += rule3Vector * rule3Scale;
  }
  
  // Apply the velocity change
  glm::vec3 newVel = vel1[index] + velChange;
  
  // Clamp the speed to maximum
  float speed = glm::length(newVel);
  if (speed > maxSpeed) {
    newVel = (newVel / speed) * maxSpeed;
  }
  
  // Store the new velocity
  vel2[index] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // Get thread index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // Direct access to boid data - positions and velocities are already sorted by grid cell
  glm::vec3 boidPos = pos[index];
  glm::vec3 velChange(0.0f);
  
  // Calculate the grid cell this boid is in
  glm::vec3 relPos = boidPos - gridMin;
  int gridX = min(max(0, (int)(relPos.x * inverseCellWidth)), gridResolution - 1);
  int gridY = min(max(0, (int)(relPos.y * inverseCellWidth)), gridResolution - 1);
  int gridZ = min(max(0, (int)(relPos.z * inverseCellWidth)), gridResolution - 1);
  
  // Variables for collecting rule information
  glm::vec3 rule1Vector(0.0f); // Cohesion
  glm::vec3 rule2Vector(0.0f); // Separation
  glm::vec3 rule3Vector(0.0f); // Alignment
  int rule1Count = 0;
  int rule3Count = 0;
  
  // Optimized z, y, x order for memory coherence with sorted data
  for (int zOffset = -1; zOffset <= 1; zOffset++) {
    int neighborZ = gridZ + zOffset;
    if (neighborZ < 0 || neighborZ >= gridResolution) continue;
    
    for (int yOffset = -1; yOffset <= 1; yOffset++) {
      int neighborY = gridY + yOffset;
      if (neighborY < 0 || neighborY >= gridResolution) continue;
      
      for (int xOffset = -1; xOffset <= 1; xOffset++) {
        int neighborX = gridX + xOffset;
        if (neighborX < 0 || neighborX >= gridResolution) continue;
        
        // Get the grid cell index
        int gridIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);
        
        // Get the start and end indices for this cell in the sorted arrays
        int cellStart = gridCellStartIndices[gridIndex];
        int cellEnd = gridCellEndIndices[gridIndex];
        
        // Skip empty cells
        if (cellStart == -1) continue;
        
        // Iterate over all boids in this cell - direct access, no indirection
        for (int i = cellStart; i < cellEnd; i++) {
          // Skip self - direct comparison of indices works since data is rearranged
          if (i == index) continue;
          
          // Direct access to position and velocity data
          glm::vec3 otherPos = pos[i];
          glm::vec3 diff = otherPos - boidPos;
          float distSq = glm::dot(diff, diff); // Squared distance - eliminates sqrt calculation
          
          // Rule 2: Separation - avoid crowding neighbors
          if (distSq < rule2DistanceSq) {
            rule2Vector -= diff; // Using diff directly rather than recomputing direction
          }
          
          // Rule 1: Cohesion - fly towards center of mass
          if (distSq < rule1DistanceSq) {
            rule1Vector += otherPos;
            rule1Count++;
          }
          
          // Rule 3: Alignment - match velocity with nearby boids
          if (distSq < rule3DistanceSq) {
            rule3Vector += vel1[i]; // Direct access
            rule3Count++;
          }
        }
      }
    }
  }
  
  // Apply Rule 2 (Separation)
  velChange += rule2Vector * rule2Scale;
  
  // Apply Rule 1 (Cohesion)
  if (rule1Count > 0) {
    rule1Vector /= rule1Count;
    velChange += (rule1Vector - boidPos) * rule1Scale;
  }
  
  // Apply Rule 3 (Alignment)
  if (rule3Count > 0) {
    rule3Vector /= rule3Count;
    velChange += rule3Vector * rule3Scale;
  }
  
  // Apply the velocity change
  glm::vec3 newVel = vel1[index] + velChange;
  
  // Clamp the speed to maximum
  float speed = glm::length(newVel);
  if (speed > maxSpeed) {
    newVel = (newVel / speed) * maxSpeed;
  }
  
  // Store the new velocity
  vel2[index] = newVel;
}

__global__ void kernRearrangeBoidData(
  int N, int *particleArrayIndices,
  glm::vec3 *sortedPos, glm::vec3 *pos,
  glm::vec3 *sortedVel1, glm::vec3 *vel1) {
  // Get thread index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // Get the original boid index
  int originalIndex = particleArrayIndices[index];
  
  // Rearrange positions and velocities to be coherent in memory
  sortedPos[index] = pos[originalIndex];
  sortedVel1[index] = vel1[originalIndex];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // Calculate grid size for kernel launch
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // Update velocity by using the naive approach
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, d_boidPositions, d_velocitiesA, d_velocitiesB);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  // Update positions using the newly calculated velocities
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dt, d_boidPositions, d_velocitiesB);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Ping-pong the velocity buffers for the next step
  std::swap(d_velocitiesA, d_velocitiesB);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // Calculate grid size for kernel launch
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  
  // Reset grid cell start/end indices to -1 (indicating empty cells)
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, d_gridCellStart, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer start indices failed!");
  
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, d_gridCellEnd, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer end indices failed!");

  // Label boids with their grid cell indices and original array indices
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
    d_boidPositions, d_boidArrayIdx, d_boidGridIdx);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // Sort boids by grid cell index using Thrust
  thrust::sort_by_key(d_thrust_boidGridIdx, d_thrust_boidGridIdx + numObjects, d_thrust_boidArrayIdx);
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");
  
  // Identify start/end indices of each cell in the sorted arrays
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, d_boidGridIdx, d_gridCellStart, d_gridCellEnd);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // Update velocities using grid-based neighbor search
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
    d_gridCellStart, d_gridCellEnd, d_boidArrayIdx,
    d_boidPositions, d_velocitiesA, d_velocitiesB);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // Update positions using the newly calculated velocities
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dt, d_boidPositions, d_velocitiesB);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Ping-pong the velocity buffers for the next step
  std::swap(d_velocitiesA, d_velocitiesB);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // Calculate grid size for kernel launch
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  
  // Reset grid cell start/end indices to -1 (indicating empty cells)
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, d_gridCellStart, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer start indices failed!");
  
  kernResetIntBuffer<<<fullBlocksPerGrid, blockSize>>>(gridCellCount, d_gridCellEnd, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer end indices failed!");

  // Step 1: Label boids with their grid cell indices and original array indices
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
    d_boidPositions, d_boidArrayIdx, d_boidGridIdx);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // Step 2: Sort boids by grid cell index using Thrust
  thrust::sort_by_key(d_thrust_boidGridIdx, d_thrust_boidGridIdx + numObjects, d_thrust_boidArrayIdx);
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");
  
  // Step 3: Identify start/end indices of each cell in the sorted arrays
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, d_boidGridIdx, d_gridCellStart, d_gridCellEnd);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // Step 4: Rearrange boid data (positions and velocities) to be coherent in memory
  kernRearrangeBoidData<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, d_boidArrayIdx, 
    d_sortedPositions, d_boidPositions,
    d_sortedVelocitiesA, d_velocitiesA);
  checkCUDAErrorWithLine("kernRearrangeBoidData failed!");

  // Step 5: Update velocities using coherent grid-based neighbor search
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
    d_gridCellStart, d_gridCellEnd, 
    d_sortedPositions, d_sortedVelocitiesA, d_sortedVelocitiesB);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

  // Step 6: Update positions using the newly calculated velocities
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dt, d_sortedPositions, d_sortedVelocitiesB);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Step 7: Ping-pong the velocity buffers for the next step
  std::swap(d_sortedVelocitiesA, d_sortedVelocitiesB);
  
  // Step 8: Swap main pointers with sorted data instead of copying
  // This eliminates expensive memory transfers for smoother visualization
  std::swap(d_boidPositions, d_sortedPositions);
  std::swap(d_velocitiesA, d_sortedVelocitiesA);
}

void Boids::endSimulation() {
  // Free basic boid buffers
  cudaFree(d_boidPositions);
  cudaFree(d_velocitiesA);
  cudaFree(d_velocitiesB);

  // Free grid-related buffers
  cudaFree(d_boidArrayIdx);
  cudaFree(d_boidGridIdx);
  cudaFree(d_gridCellStart);
  cudaFree(d_gridCellEnd);

  // Free coherent grid buffers
  cudaFree(d_sortedPositions);
  cudaFree(d_sortedVelocitiesA);
  cudaFree(d_sortedVelocitiesB);
}

void Boids::unitTest() {
  // Modified to use size 12 instead of size 10
  int *dev_intKeys;
  int *dev_intValues;
  int N = 12;  // Changed from 10 to 12

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0;  intValues[0] = 0;
  intKeys[1] = 1;  intValues[1] = 1;
  intKeys[2] = 0;  intValues[2] = 2;
  intKeys[3] = 3;  intValues[3] = 3;
  intKeys[4] = 0;  intValues[4] = 4;
  intKeys[5] = 2;  intValues[5] = 5;
  intKeys[6] = 2;  intValues[6] = 6;
  intKeys[7] = 0;  intValues[7] = 7;
  intKeys[8] = 5;  intValues[8] = 8;
  intKeys[9] = 6;  intValues[9] = 9;
  intKeys[10] = 4; intValues[10] = 10;  // Added two new entries
  intKeys[11] = 1; intValues[11] = 11;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
