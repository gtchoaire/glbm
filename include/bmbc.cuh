#ifndef GLBM_BMBC_CUH
#define GLBM_BMBC_CUH

#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include "globalStructs.cuh"
#include "localFieldStatistics.cuh"

enum BMBCLine
{
    BMBC_Y_CENTER = 0,
    BMBC_X_CENTER,
    BMBC_LINES
};

__host__ __device__ constexpr std::size_t bmbcLineSize(const int line)
{
    return line == BMBC_X_CENTER ? GLOBAL_NX : GLOBAL_NY;
}

__host__ __device__ constexpr std::size_t bmbcLineNodeOffset(const int line)
{
    std::size_t offset = 0;
    for (int previous = 0; previous < line; ++previous)
        offset += bmbcLineSize(previous);
    return offset;
}

__host__ __device__ constexpr std::size_t bmbcPoint(
    const int line, const int field, const std::size_t index)
{
    return NUM_LOCAL_FIELDS * bmbcLineNodeOffset(line) +
           field * bmbcLineSize(line) + index;
}

constexpr std::size_t BMBC_LINE_NODES = bmbcLineNodeOffset(BMBC_LINES);
constexpr std::size_t BMBC_PROFILE_POINTS = NUM_LOCAL_FIELDS * BMBC_LINE_NODES;

struct BMBCEstimate
{
    double mean = 0.0;
    double error = 0.0;
    std::size_t group = 0;
    std::size_t groups = 0;
};

struct BMBCProfiles
{
    double *d_batch_sum = nullptr;   // One running sum per profile point.
    double *d_batch_means = nullptr; // [completed block][profile point].

    std::size_t block_samples = 0;
    std::size_t capacity_blocks = 0;
    std::size_t samples_in_block = 0;
    std::size_t completed_blocks = 0;
    std::size_t total_samples = 0;
};

// Per-device partial sum of the profile points owned by each subdomain.
// Used only with NUM_GPUS == 2. Owns memory on its associated device.
struct BMBCLocalProfileSum
{
    double *d_sum = nullptr;
};

// Stage 1 of dual-GPU BMBC: extraction only. No inter-device transfer yet.
// Must be called with the owning CUDA device selected. The caller must
// wait for the LBM state to be ready in the supplied stream.
void allocateBMBCLocalProfileSum(BMBCLocalProfileSum &local);
void accumulateBMBCLocalProfileSum(BMBCLocalProfileSum &local, const LBMState &state, const SubDomainInfo &info, cudaStream_t stream);
void freeBMBCLocalProfileSum(BMBCLocalProfileSum &local);

// Dual-GPU transfer staging. GPU 0 owns d_remote; h_remote is pinned host
// memory. The partial block sums remain on their respective GPUs until
// block completion or a checkpoint is requested.
struct BMBCMultiGPUExchange
{
    double *h_remote = nullptr;
    double *d_remote = nullptr;
    cudaEvent_t remote_ready = nullptr;
    cudaEvent_t remote_consumed = nullptr;
    int owner_device = -1;
    int remote_device = -1;
    std::size_t pending_samples = 0;
};

void allocateBMBCMultiGPUExchange(BMBCMultiGPUExchange &exchange, int owner_device, int remote_device);
void flushBMBCMultiGPUProfiles(BMBCProfiles &monitor, BMBCLocalProfileSum &local0, BMBCLocalProfileSum &local1, BMBCMultiGPUExchange &exchange, int device0, int device1, cudaStream_t stream0, cudaStream_t stream1);
void recordBMBCMultiGPUSample(BMBCProfiles &monitor, BMBCLocalProfileSum &local0, BMBCLocalProfileSum &local1, BMBCMultiGPUExchange &exchange, int device0, int device1, cudaStream_t stream0, cudaStream_t stream1);
void freeBMBCMultiGPUExchange(BMBCMultiGPUExchange &exchange);

// Allocate in the currently selected CUDA device.
void allocateBMBCProfiles(BMBCProfiles &monitor,
                          std::size_t block_samples,
                          std::size_t capacity_blocks);

// Call once per statistics sample, in the same stream as the completed LBM step.
void accumulateBMBCProfiles(BMBCProfiles &monitor,
                            const LBMState &state,
                            const SubDomainInfo &info,
                            cudaStream_t stream);

// Free in the device where the monitor was allocated.
// Stage 2: retrieve completed block means for independent validation.
void copyBMBCCompletedMeansToHost(
    const BMBCProfiles &monitor, std::vector<double> &host_means);

void freeBMBCProfiles(BMBCProfiles &monitor);

std::size_t usableBMBCBlocks(std::size_t count);

BMBCEstimate estimateBMBC(const std::vector<double> &series);

void reportBMBCProfiles(
    const BMBCProfiles &monitor,
    std::size_t completed_steps,
    cudaStream_t stream);

#endif
