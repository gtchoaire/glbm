#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "model.h"
#include "index.h"
#include <stdexcept>
#include <string>

/* --------------------------- CONSTANTS --------------------------- */

constexpr dfloat OMEGA = 1.0 / TAU;            // (tau)^-1
constexpr dfloat OMEGAd2 = OMEGA / 2.0;        // OMEGA/2
constexpr dfloat T_OMEGA = 1.0 - OMEGA;        // 1-OMEGA

#ifndef BLOCK_NX_VALUE
#define BLOCK_NX_VALUE 32
#endif
#ifndef BLOCK_NY_VALUE
#define BLOCK_NY_VALUE 4
#endif
#ifndef BLOCK_NZ_VALUE
#define BLOCK_NZ_VALUE 2
#endif

// CUDA execution geometry. It can be tuned without changing the logical AB
// tile below, which is part of the validated numerical method.
constexpr int BLOCK_NX = BLOCK_NX_VALUE;
constexpr int BLOCK_NY = BLOCK_NY_VALUE;
constexpr int BLOCK_NZ = BLOCK_NZ_VALUE;
static_assert(BLOCK_NX > 0 && BLOCK_NY > 0 && BLOCK_NZ > 0, "CUDA block dimensions must be positive");
static_assert(BLOCK_NX * BLOCK_NY * BLOCK_NZ <= 1024, "CUDA block cannot exceed 1024 threads");
static_assert(NX % BLOCK_NX == 0 && NY % BLOCK_NY == 0 && NZ % BLOCK_NZ == 0,
              "Domain dimensions must be divisible by CUDA block dimensions");

// Logical tile used by the validated AB first-step reconstruction.  Keep it
// independent from the CUDA block geometry so launch tuning does not alter
// which neighbouring states are collided during that first step.
constexpr int AB_TILE_NX = 8;
constexpr int AB_TILE_NY = 8;
constexpr int AB_TILE_NZ = 4;

constexpr size_t BLOCK_LBM_SIZE = BLOCK_NX * BLOCK_NY * BLOCK_NZ;

constexpr size_t NUM_BLOCK_X = NX / BLOCK_NX;
constexpr size_t NUM_BLOCK_Y = NY / BLOCK_NY;
constexpr size_t NUM_BLOCK_Z = NZ / BLOCK_NZ;

constexpr size_t NUM_BLOCK = NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;

constexpr size_t NUMBER_LBM_NODES = NUM_BLOCK * BLOCK_LBM_SIZE;
constexpr size_t MEM_SIZE_MOM = sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS;

#ifndef SAVE_VTI_OUTPUT_VALUE
#define SAVE_VTI_OUTPUT_VALUE 1
#endif

constexpr bool SAVE_VTI_OUTPUT = SAVE_VTI_OUTPUT_VALUE != 0;

#ifndef SAVE_BINARY_OUTPUT_VALUE
#define SAVE_BINARY_OUTPUT_VALUE 0
#endif

constexpr bool SAVE_BINARY_OUTPUT = SAVE_BINARY_OUTPUT_VALUE != 0;

#ifndef BINARY_OUTPUT_INTERVAL
#define BINARY_OUTPUT_INTERVAL 1000
#endif

constexpr size_t BINARY_OUTPUT_INTERVAL_VALUE = BINARY_OUTPUT_INTERVAL;

#ifndef SAVE_KINETIC_ENERGY_OUTPUT_VALUE
#define SAVE_KINETIC_ENERGY_OUTPUT_VALUE 0
#endif

constexpr bool SAVE_KINETIC_ENERGY_OUTPUT = SAVE_KINETIC_ENERGY_OUTPUT_VALUE != 0;

#ifndef KINETIC_ENERGY_OUTPUT_INTERVAL
#define KINETIC_ENERGY_OUTPUT_INTERVAL 1000
#endif

constexpr size_t KINETIC_ENERGY_INTERVAL = KINETIC_ENERGY_OUTPUT_INTERVAL;

#ifndef PRINT_INTERVAL_VALUE
#define PRINT_INTERVAL_VALUE 10000
#endif

constexpr int PRINT_INTERVAL = PRINT_INTERVAL_VALUE;
static_assert(PRINT_INTERVAL > 0, "PRINT_INTERVAL must be positive");

#ifndef CHECKPOINT_INTERVAL_VALUE
#define CHECKPOINT_INTERVAL_VALUE 0
#endif
#ifndef CHECKPOINT_PATH_VALUE
#define CHECKPOINT_PATH_VALUE "checkpoint.glbm"
#endif
#ifndef RESTART_PATH_VALUE
#define RESTART_PATH_VALUE ""
#endif

constexpr size_t CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL_VALUE;
constexpr const char *CHECKPOINT_PATH = CHECKPOINT_PATH_VALUE;
constexpr const char *RESTART_PATH = RESTART_PATH_VALUE;
constexpr bool CHECKPOINT_ENABLED = CHECKPOINT_INTERVAL > 0;
constexpr bool RESTART_ENABLED = RESTART_PATH_VALUE[0] != '\0';

#define checkCuda(cmd) do { \
    const cudaError_t status = (cmd); \
    if (status != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status)); \
} while (0)

#endif
