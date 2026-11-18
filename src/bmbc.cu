#include "../include/bmbc.cuh"
#include "../include/globalFunctions.cuh"

#include <limits>
#include <stdexcept>
#include <cstdio>

#include PROPERTIES

namespace
{
    __device__ __forceinline__ std::size_t scalarIndex3D(
        const int x, const int y, const int z)
    {
        return idxScalarBlock(
            x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
            x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
    }

    __device__ __forceinline__ double physicalVelocity(
        const LBMState &state, const int component,
        const int x, const int y, const int z)
    {
        return static_cast<double>(
                   state.device(component, scalarIndex3D(x, y, z))) /
               static_cast<double>(F_M_I_SCALE);
    }

    __global__ void gpuAccumulateBMBCProfiles(
        const LBMState state, double *batch_sum)
    {
        const std::size_t p =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (p >= BMBC_PROFILE_POINTS)
            return;

        // For odd sizes, the lower and upper center indices are identical.
        const int x0 = (NX - 1) / 2;
        const int x1 = NX / 2;
        const int y0 = (NY - 1) / 2;
        const int y1 = NY / 2;
        const int z0 = (NZ - 1) / 2;
        const int z1 = NZ / 2;

        double value;
        if (p < static_cast<std::size_t>(NY))
        {
            // ux(y) at x = (NX-1)/2, z = (NZ-1)/2.
            const int y = static_cast<int>(p);
            value = 0.25 * (physicalVelocity(state, M_UX_INDEX, x0, y, z0) +
                            physicalVelocity(state, M_UX_INDEX, x1, y, z0) +
                            physicalVelocity(state, M_UX_INDEX, x0, y, z1) +
                            physicalVelocity(state, M_UX_INDEX, x1, y, z1));
        }
        else
        {
            // uy(x) at y = (NY-1)/2, z = (NZ-1)/2.
            const int x = static_cast<int>(p - NY);
            value = 0.25 * (physicalVelocity(state, M_UY_INDEX, x, y0, z0) +
                            physicalVelocity(state, M_UY_INDEX, x, y1, z0) +
                            physicalVelocity(state, M_UY_INDEX, x, y0, z1) +
                            physicalVelocity(state, M_UY_INDEX, x, y1, z1));
        }

        // One thread writes each point; no atomics are needed.
        batch_sum[p] += value;
    }

    __global__ void gpuCompleteBMBCBlock(
        double *batch_sum, double *batch_means,
        const std::size_t completed_blocks, const double inverse_block_samples)
    {
        const std::size_t p =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (p >= BMBC_PROFILE_POINTS)
            return;

        batch_means[completed_blocks * BMBC_PROFILE_POINTS + p] =
            batch_sum[p] * inverse_block_samples;
        batch_sum[p] = 0.0;
    }
    // One thread handles an entire time series at one profile point.
    // All quantities are accumulated in double precision.

    __global__ void gpuComputeBMBCStatistics(
        const double *batch_means,
        const std::size_t completed_blocks,
        const std::size_t group_size,
        double *statistics)
    {
        const std::size_t p =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (p >= BMBC_PROFILE_POINTS)
            return;

        const std::size_t stride = BMBC_PROFILE_POINTS;
        const std::size_t K = completed_blocks / group_size;

        // Mean of one group of consecutive basic blocks.
        // Every basic block contains the same number of samples.
        auto groupedMean = [&](const std::size_t group)
        {
            double sum = 0.0;

            for (std::size_t j = 0; j < group_size; ++j)
            {
                const std::size_t k = group * group_size + j;
                sum += batch_means[k * stride + p];
            }

            return sum / static_cast<double>(group_size);
        };

        // Mean of all complete groups.
        double sum = 0.0;

        for (std::size_t k = 0; k < K; ++k)
            sum += groupedMean(k);

        const double mean = sum / static_cast<double>(K);

        // BMBC terms.
        double s0 = 0.0;
        double s1 = 0.0;
        double previous = 0.0;

        for (std::size_t k = 0; k < K; ++k)
        {
            const double centered = groupedMean(k) - mean;

            s0 += centered * centered;

            if (k > 0)
                s1 += previous * centered;

            previous = centered;
        }

        const double n = static_cast<double>(K);

        const double variance =
            (s0 + 2.0 * s1) / ((n - 1.0) * (n - 2.0));

        statistics[BMBC_STAT_MEAN * stride + p] = mean;
        statistics[BMBC_STAT_S0 * stride + p] = s0;
        statistics[BMBC_STAT_S1 * stride + p] = s1;
        statistics[BMBC_STAT_VARIANCE * stride + p] = variance;
    }

} // namespace

void allocateBMBCProfiles(BMBCProfiles &monitor,
                          const std::size_t block_samples,
                          const std::size_t capacity_blocks)
{
    if constexpr (NUM_GPUS != 1)
        throw std::runtime_error(
            "BMBC centerline extraction currently supports one GPU only");
    if (monitor.d_batch_sum || monitor.d_batch_means || monitor.d_statistics)
        throw std::runtime_error("BMBC monitor already allocated");
    if (block_samples == 0 || capacity_blocks == 0)
        throw std::runtime_error("BMBC sizes must be positive");
    if (capacity_blocks > std::numeric_limits<std::size_t>::max() /
                              (BMBC_PROFILE_POINTS * sizeof(double)))
        throw std::runtime_error("BMBC allocation size overflow");

    const std::size_t sum_bytes = BMBC_PROFILE_POINTS * sizeof(double);
    const std::size_t history_bytes = capacity_blocks * sum_bytes;
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&monitor.d_batch_sum),
                         sum_bytes));
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&monitor.d_batch_means),
                         history_bytes));
    checkCuda(cudaMemset(monitor.d_batch_sum, 0, sum_bytes));
    checkCuda(cudaMalloc(
        reinterpret_cast<void **>(&monitor.d_statistics),
        BMBC_STAT_FIELDS * sum_bytes));

    monitor.block_samples = block_samples;
    monitor.capacity_blocks = capacity_blocks;
    monitor.samples_in_block = 0;
    monitor.completed_blocks = 0;
    monitor.total_samples = 0;
}

void computeBMBCGroupedStatistics(
    BMBCProfiles &monitor,
    const std::size_t group_size,
    const cudaStream_t stream)
{
    if (!monitor.d_batch_means || !monitor.d_statistics)
        throw std::runtime_error("BMBC monitor is not allocated");

    if (group_size == 0)
        throw std::runtime_error("BMBC group size must be positive");

    const std::size_t K =
        monitor.completed_blocks / group_size;

    if (K < 3)
        throw std::runtime_error(
            "BMBC requires at least three complete groups");

    constexpr int threads = 256;

    const unsigned int blocks =
        static_cast<unsigned int>(
            (BMBC_PROFILE_POINTS + threads - 1) / threads);

    gpuComputeBMBCStatistics<<<blocks, threads, 0, stream>>>(
        monitor.d_batch_means,
        monitor.completed_blocks,
        group_size,
        monitor.d_statistics);

    checkCuda(cudaGetLastError());
}

void computeBMBCStatistics(
    BMBCProfiles &monitor,
    const cudaStream_t stream)
{
    computeBMBCGroupedStatistics(monitor, 1, stream);
}

static void ensureBMBCBlockCapacity(
    BMBCProfiles &monitor,
    const cudaStream_t stream)
{
    if (monitor.completed_blocks < monitor.capacity_blocks)
        return;

    const std::size_t bytes_per_block =
        BMBC_PROFILE_POINTS * sizeof(double);

    const std::size_t max_blocks =
        std::numeric_limits<std::size_t>::max() /
        bytes_per_block;

    if (monitor.capacity_blocks >= max_blocks)
        throw std::runtime_error(
            "BMBC storage capacity overflow");

    // Double the capacity whenever the storage becomes full.
    const std::size_t new_capacity =
        monitor.capacity_blocks > max_blocks / 2
            ? max_blocks
            : monitor.capacity_blocks * 2;

    double *new_batch_means = nullptr;

    checkCuda(cudaMalloc(
        reinterpret_cast<void **>(&new_batch_means),
        new_capacity * bytes_per_block));

    // Preserve all previously completed blocks.
    const std::size_t used_bytes =
        monitor.completed_blocks * bytes_per_block;

    if (used_bytes > 0)
    {
        checkCuda(cudaMemcpyAsync(
            new_batch_means,
            monitor.d_batch_means,
            used_bytes,
            cudaMemcpyDeviceToDevice,
            stream));

        checkCuda(cudaStreamSynchronize(stream));
    }

    checkCuda(cudaFree(monitor.d_batch_means));

    monitor.d_batch_means = new_batch_means;
    monitor.capacity_blocks = new_capacity;
}

void accumulateBMBCProfiles(BMBCProfiles &monitor,
                            const LBMState &state,
                            const cudaStream_t stream)
{
    if (!monitor.d_batch_sum || !monitor.d_batch_means)
        throw std::runtime_error("BMBC monitor is not allocated");

    const bool complete = monitor.samples_in_block + 1 == monitor.block_samples;

    if (complete)
        ensureBMBCBlockCapacity(monitor, stream);

    constexpr int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((BMBC_PROFILE_POINTS + threads - 1) / threads);

    gpuAccumulateBMBCProfiles<<<blocks, threads, 0, stream>>>(
        state, monitor.d_batch_sum);
    checkCuda(cudaGetLastError());

    ++monitor.samples_in_block;
    ++monitor.total_samples;

    if (complete)
    {
        gpuCompleteBMBCBlock<<<blocks, threads, 0, stream>>>(
            monitor.d_batch_sum, monitor.d_batch_means,
            monitor.completed_blocks,
            1.0 / static_cast<double>(monitor.block_samples));
        checkCuda(cudaGetLastError());
        ++monitor.completed_blocks;
        monitor.samples_in_block = 0;
    }
}

void copyBMBCCompletedMeansToHost(
    const BMBCProfiles &monitor, std::vector<double> &host_means)
{
    if (monitor.d_batch_means == nullptr)
        throw std::runtime_error("BMBC monitor is not allocated");

    const std::size_t count = monitor.completed_blocks * BMBC_PROFILE_POINTS;
    host_means.resize(count);
    if (count > 0)
        checkCuda(cudaMemcpy(host_means.data(), monitor.d_batch_means,
                             count * sizeof(double), cudaMemcpyDeviceToHost));
}

void copyBMBCStatisticsToHost(
    const BMBCProfiles &monitor, std::vector<double> &host_statistics)
{
    if (!monitor.d_statistics)
        throw std::runtime_error("BMBC monitor is not allocated");
    if (monitor.completed_blocks < 3)
        throw std::runtime_error("Not enough completed BMBC blocks");

    const std::size_t count = BMBC_STAT_FIELDS * BMBC_PROFILE_POINTS;
    host_statistics.resize(count);
    checkCuda(cudaMemcpy(host_statistics.data(), monitor.d_statistics,
                         count * sizeof(double), cudaMemcpyDeviceToHost));
}

void reportBMBCProfiles(
    BMBCProfiles &monitor,
    const std::size_t completed_steps,
    const cudaStream_t stream)
{
    if (monitor.completed_blocks < 3)
        return;

    // Compute statistics for every profile point on the GPU.
    computeBMBCGroupedStatistics(monitor, 1, stream);

    checkCuda(cudaStreamSynchronize(stream));

    std::vector<double> statistics;
    copyBMBCStatisticsToHost(monitor, statistics);

    const std::size_t stride = BMBC_PROFILE_POINTS;

    const auto printPoint = [&](const std::size_t p, const char *name)
    {
        const double mean =
            statistics[BMBC_STAT_MEAN * stride + p];

        const double s0 =
            statistics[BMBC_STAT_S0 * stride + p];

        const double s1 =
            statistics[BMBC_STAT_S1 * stride + p];

        const double variance =
            statistics[BMBC_STAT_VARIANCE * stride + p];

        std::printf(
            "  %s | Mean %.9e | S1/S0 %.6f | Variance %.9e\n",
            name,
            mean,
            s0 > 0.0 ? s1 / s0 : 0.0,
            variance);
    };

    std::printf(
        "\nBMBC | Step %zu | Samples %zu | Blocks %zu | M %zu\n",
        completed_steps,
        monitor.total_samples,
        monitor.completed_blocks,
        monitor.block_samples);

    printPoint(NY - 2, "ux(y=NY-2)");
    printPoint(NY + NX - 2, "uy(x=NX-2)");
}

void freeBMBCProfiles(BMBCProfiles &monitor)
{
    if (monitor.d_batch_sum)
        checkCuda(cudaFree(monitor.d_batch_sum));
    if (monitor.d_batch_means)
        checkCuda(cudaFree(monitor.d_batch_means));
    if (monitor.d_statistics)
        checkCuda(cudaFree(monitor.d_statistics));
    monitor = BMBCProfiles{};
}
