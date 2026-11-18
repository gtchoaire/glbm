#ifndef GLBM_BMBC_CUH
#define GLBM_BMBC_CUH

#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include "globalStructs.cuh"

// Profile 0: ux(y) at the geometric center (x,z).
// Profile 1: uy(x) at the geometric center (y,z).
// Stored contiguously in this order.
constexpr std::size_t BMBC_PROFILE_POINTS =
    static_cast<std::size_t>(NY) + static_cast<std::size_t>(NX);

// Four statistics for each profile point, stored field-major.
constexpr std::size_t BMBC_STAT_FIELDS = 4;
enum BMBCStatField : int
{
    BMBC_STAT_MEAN = 0,
    BMBC_STAT_S0,
    BMBC_STAT_S1,
    BMBC_STAT_VARIANCE
};

struct BMBCProfiles
{
    double *d_batch_sum = nullptr;   // One running sum per profile point.
    double *d_batch_means = nullptr; // [completed block][profile point].
    double *d_statistics = nullptr;  // [statistic][profile point].

    std::size_t block_samples = 0;
    std::size_t capacity_blocks = 0;
    std::size_t samples_in_block = 0;
    std::size_t completed_blocks = 0;
    std::size_t total_samples = 0;
};

// Allocate in the currently selected CUDA device.
void allocateBMBCProfiles(BMBCProfiles &monitor,
                          std::size_t block_samples,
                          std::size_t capacity_blocks);

// Call once per statistics sample, in the same stream as the completed LBM step.
void accumulateBMBCProfiles(BMBCProfiles &monitor,
                            const LBMState &state,
                            cudaStream_t stream);

// Free in the device where the monitor was allocated.
// Stage 2: retrieve completed block means for independent validation.
void copyBMBCCompletedMeansToHost(
    const BMBCProfiles &monitor, std::vector<double> &host_means);

// Stage 3: calculate BMBC estimate for each point using completed blocks.
// Requires at least three completed blocks. Ignores a partially filled block.
void computeBMBCStatistics(BMBCProfiles &monitor, cudaStream_t stream);

// Caller must synchronize computeBMBCStatistics's stream before copying.
void copyBMBCStatisticsToHost(
    const BMBCProfiles &monitor, std::vector<double> &host_statistics);

void freeBMBCProfiles(BMBCProfiles &monitor);

void computeBMBCGroupedStatistics(
    BMBCProfiles &monitor,
    std::size_t group_size,
    cudaStream_t stream);

void reportBMBCProfiles(
    BMBCProfiles &monitor,
    std::size_t completed_steps,
    cudaStream_t stream);

#endif
