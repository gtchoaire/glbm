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

    __device__ __forceinline__ double yProfileMoment(
        const int field, const double ux, const double uy)
    {
        if (field == 0) return ux;
        if (field == 1) return uy;
        if (field == 2) return ux * ux;
        return ux * uy;
    }

    __device__ __forceinline__ double xProfileMoment(
        const int field, const double ux, const double uy)
    {
        if (field == 0) return ux;
        if (field == 1) return uy;
        if (field == 2) return uy * uy;
        return ux * uy;
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

        double value = 0.0;
        if constexpr (BMBC_TURBULENCE_PROFILES)
        {
            const std::size_t y_values = 4 * static_cast<std::size_t>(NY);
            if (p < y_values)
            {
                const int field = static_cast<int>(p / NY);
                const int y = static_cast<int>(p % NY);
                const int xs[2]{x0, x1};
                const int zs[2]{z0, z1};
                for (int iz = 0; iz < 2; ++iz)
                    for (int ix = 0; ix < 2; ++ix)
                    {
                        const double ux = physicalVelocity(
                            state, M_UX_INDEX, xs[ix], y, zs[iz]);
                        const double uy = physicalVelocity(
                            state, M_UY_INDEX, xs[ix], y, zs[iz]);
                        value += 0.25 * yProfileMoment(field, ux, uy);
                    }
            }
            else
            {
                const std::size_t local = p - y_values;
                const int field = static_cast<int>(local / NX);
                const int x = static_cast<int>(local % NX);
                const int ys[2]{y0, y1};
                const int zs[2]{z0, z1};
                for (int iz = 0; iz < 2; ++iz)
                    for (int iy = 0; iy < 2; ++iy)
                    {
                        const double ux = physicalVelocity(
                            state, M_UX_INDEX, x, ys[iy], zs[iz]);
                        const double uy = physicalVelocity(
                            state, M_UY_INDEX, x, ys[iy], zs[iz]);
                        value += 0.25 * xProfileMoment(field, ux, uy);
                    }
            }
        }
        else
        {
            if (p < static_cast<std::size_t>(NY))
            {
                const int y = static_cast<int>(p);
                value = 0.25 * (physicalVelocity(state, M_UX_INDEX, x0, y, z0) +
                                physicalVelocity(state, M_UX_INDEX, x1, y, z0) +
                                physicalVelocity(state, M_UX_INDEX, x0, y, z1) +
                                physicalVelocity(state, M_UX_INDEX, x1, y, z1));
            }
            else
            {
                const int x = static_cast<int>(p - NY);
                value = 0.25 * (physicalVelocity(state, M_UY_INDEX, x, y0, z0) +
                                physicalVelocity(state, M_UY_INDEX, x, y1, z0) +
                                physicalVelocity(state, M_UY_INDEX, x, y0, z1) +
                                physicalVelocity(state, M_UY_INDEX, x, y1, z1));
            }
        }

        // One thread writes each point; no atomics are needed.
        batch_sum[p] += value;
    }

    // For a two-way Z decomposition of an even-sized global domain, the
    // two geometric-center planes belong to different devices. Each GPU
    // accumulates HALF of the interpolated central profile. Adding both
    // partial sums later recovers the full instantaneous-profile sum.
    __global__ void gpuAccumulateBMBCLocalProfile(
        const LBMState state, const int local_center_z, double *partial_sum)
    {
        const std::size_t p =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (p >= BMBC_PROFILE_POINTS)
            return;

        const int x0 = (GLOBAL_NX - 1) / 2;
        const int x1 = GLOBAL_NX / 2;
        const int y0 = (GLOBAL_NY - 1) / 2;
        const int y1 = GLOBAL_NY / 2;
        double contribution = 0.0;

        if constexpr (BMBC_TURBULENCE_PROFILES)
        {
            const std::size_t y_values =
                4 * static_cast<std::size_t>(GLOBAL_NY);
            if (p < y_values)
            {
                const int field = static_cast<int>(p / GLOBAL_NY);
                const int y = static_cast<int>(p % GLOBAL_NY);
                const int xs[2]{x0, x1};
                for (int ix = 0; ix < 2; ++ix)
                {
                    const double ux = physicalVelocity(
                        state, M_UX_INDEX, xs[ix], y, local_center_z);
                    const double uy = physicalVelocity(
                        state, M_UY_INDEX, xs[ix], y, local_center_z);
                    contribution += 0.25 * yProfileMoment(field, ux, uy);
                }
            }
            else
            {
                const std::size_t local = p - y_values;
                const int field = static_cast<int>(local / GLOBAL_NX);
                const int x = static_cast<int>(local % GLOBAL_NX);
                const int ys[2]{y0, y1};
                for (int iy = 0; iy < 2; ++iy)
                {
                    const double ux = physicalVelocity(
                        state, M_UX_INDEX, x, ys[iy], local_center_z);
                    const double uy = physicalVelocity(
                        state, M_UY_INDEX, x, ys[iy], local_center_z);
                    contribution += 0.25 * xProfileMoment(field, ux, uy);
                }
            }
        }
        else
        {
            if (p < static_cast<std::size_t>(GLOBAL_NY))
            {
                const int y = static_cast<int>(p);
                contribution = 0.25 * (
                    physicalVelocity(state, M_UX_INDEX, x0, y, local_center_z) +
                    physicalVelocity(state, M_UX_INDEX, x1, y, local_center_z));
            }
            else
            {
                const int x = static_cast<int>(p - GLOBAL_NY);
                contribution = 0.25 * (
                    physicalVelocity(state, M_UY_INDEX, x, y0, local_center_z) +
                    physicalVelocity(state, M_UY_INDEX, x, y1, local_center_z));
            }
        }

        partial_sum[p] += contribution;
    }

    __global__ void gpuMergeBMBCLocalProfiles(
        double *global_sum, double *local_sum, const double *remote_sum)
    {
        const std::size_t p =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (p >= BMBC_PROFILE_POINTS)
            return;
        global_sum[p] += local_sum[p] + remote_sum[p];
        local_sum[p] = 0.0;
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
            const double value = groupedMean(k);
            const double centered = value - mean;

            s0 += centered * centered;

            // Russo & Luchini's finite-sample BMBC correction is
            // sum(X_k X_{k+1} - mean^2), not the usual centered lag-one
            // covariance.  Those expressions differ by an endpoint term
            // for a finite series.
            if (k > 0)
                s1 += previous * value - mean * mean;

            previous = value;
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

void allocateBMBCLocalProfileSum(BMBCLocalProfileSum &local)
{
    if constexpr (NUM_GPUS != 2)
        throw std::runtime_error("Local BMBC partial sums require two GPUs");
    if (local.d_sum)
        throw std::runtime_error("BMBC local profile sum already allocated");

    const std::size_t bytes = BMBC_PROFILE_POINTS * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&local.d_sum), bytes));
    checkCuda(cudaMemset(local.d_sum, 0, bytes));
}

void accumulateBMBCLocalProfileSum(BMBCLocalProfileSum &local, const LBMState &state, const SubDomainInfo &info, const cudaStream_t stream)
{
    if (!local.d_sum)
        throw std::runtime_error("BMBC local profile sum not allocated");
    if constexpr (NUM_GPUS != 2)
        throw std::runtime_error("Local BMBC partial sums require two GPUs");

    const int lower_global_z = (GLOBAL_NZ - 1) / 2;
    const int upper_global_z = GLOBAL_NZ / 2;
    const int global_center_z = info.global_z_start == 0
                                    ? lower_global_z : upper_global_z;
    if (global_center_z < info.global_z_start ||
        global_center_z > info.global_z_end)
        throw std::runtime_error("BMBC geometric center not in subdomain");
    const int local_center_z = global_center_z - info.global_z_start;

    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (BMBC_PROFILE_POINTS + threads - 1) / threads);
    gpuAccumulateBMBCLocalProfile<<<blocks, threads, 0, stream>>>(
        state, local_center_z, local.d_sum);
    checkCuda(cudaGetLastError());
}

void freeBMBCLocalProfileSum(BMBCLocalProfileSum &local)
{
    if (local.d_sum)
        checkCuda(cudaFree(local.d_sum));
    local = BMBCLocalProfileSum{};
}

static void ensureBMBCBlockCapacity(BMBCProfiles &monitor, cudaStream_t stream);

void allocateBMBCMultiGPUExchange(BMBCMultiGPUExchange &exchange, const int owner_device, const int remote_device)
{
    if constexpr (NUM_GPUS != 2)
        throw std::runtime_error("BMBC exchange requires two GPUs");
    if (exchange.h_remote || exchange.d_remote || exchange.remote_ready || exchange.remote_consumed)
        throw std::runtime_error("BMBC exchange already allocated");

    const std::size_t bytes = BMBC_PROFILE_POINTS * sizeof(double);
    exchange.owner_device = owner_device;
    exchange.remote_device = remote_device;

    checkCuda(cudaSetDevice(owner_device));
    checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&exchange.h_remote), bytes, cudaHostAllocPortable));
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&exchange.d_remote), bytes));
    checkCuda(cudaEventCreateWithFlags(&exchange.remote_consumed, cudaEventDisableTiming));
    checkCuda(cudaEventRecord(exchange.remote_consumed));

    checkCuda(cudaSetDevice(remote_device));
    checkCuda(cudaEventCreateWithFlags(&exchange.remote_ready, cudaEventDisableTiming));

    checkCuda(cudaSetDevice(owner_device));
    exchange.pending_samples = 0;
}

void flushBMBCMultiGPUProfiles(BMBCProfiles &monitor, BMBCLocalProfileSum &local0, BMBCLocalProfileSum &local1, BMBCMultiGPUExchange &exchange, const int device0, const int device1, const cudaStream_t stream0, const cudaStream_t stream1)
{
    if (exchange.pending_samples == 0)
        return;
    if (!monitor.d_batch_sum || !local0.d_sum || !local1.d_sum ||
        !exchange.h_remote || !exchange.d_remote)
        throw std::runtime_error("BMBC multi-GPU buffers not allocated");
    if (device0 != exchange.owner_device || device1 != exchange.remote_device)
        throw std::runtime_error("BMBC multi-GPU device mismatch");

    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (BMBC_PROFILE_POINTS + threads - 1) / threads);
    const std::size_t bytes = BMBC_PROFILE_POINTS * sizeof(double);

    // GPU 1 produces the host staging buffer and GPU 0 consumes it. Events
    // keep both dependencies asynchronous and prevent premature buffer reuse.
    checkCuda(cudaSetDevice(device1));
    checkCuda(cudaStreamWaitEvent(stream1, exchange.remote_consumed, 0));
    checkCuda(cudaMemcpyAsync(exchange.h_remote, local1.d_sum, bytes, cudaMemcpyDeviceToHost, stream1));
    checkCuda(cudaMemsetAsync(local1.d_sum, 0, bytes, stream1));
    checkCuda(cudaEventRecord(exchange.remote_ready, stream1));

    checkCuda(cudaSetDevice(device0));
    checkCuda(cudaStreamWaitEvent(stream0, exchange.remote_ready, 0));
    checkCuda(cudaMemcpyAsync(exchange.d_remote, exchange.h_remote, bytes, cudaMemcpyHostToDevice, stream0));
    checkCuda(cudaEventRecord(exchange.remote_consumed, stream0));
    gpuMergeBMBCLocalProfiles<<<blocks, threads, 0, stream0>>>(
        monitor.d_batch_sum, local0.d_sum, exchange.d_remote);
    checkCuda(cudaGetLastError());
    exchange.pending_samples = 0;
}

void recordBMBCMultiGPUSample(BMBCProfiles &monitor, BMBCLocalProfileSum &local0, BMBCLocalProfileSum &local1, BMBCMultiGPUExchange &exchange, const int device0, const int device1, const cudaStream_t stream0, const cudaStream_t stream1)
{
    if (!monitor.d_batch_means || !monitor.d_batch_sum)
        throw std::runtime_error("BMBC monitor not allocated");

    const bool complete = monitor.samples_in_block + 1 == monitor.block_samples;
    ++monitor.samples_in_block;
    ++monitor.total_samples;
    ++exchange.pending_samples;
    if (!complete)
        return;

    // Preserve the existing grow-on-demand behavior, on the owner GPU.
    checkCuda(cudaSetDevice(device0));
    flushBMBCMultiGPUProfiles(monitor, local0, local1, exchange, device0, device1, stream0, stream1);
    // Complete the basic block in the same layout as single-GPU BMBC.
    ensureBMBCBlockCapacity(monitor, stream0);
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (BMBC_PROFILE_POINTS + threads - 1) / threads);
    gpuCompleteBMBCBlock<<<blocks, threads, 0, stream0>>>(
        monitor.d_batch_sum, monitor.d_batch_means,
        monitor.completed_blocks,
        1.0 / static_cast<double>(monitor.block_samples));
    checkCuda(cudaGetLastError());
    ++monitor.completed_blocks;
    monitor.samples_in_block = 0;
}

void freeBMBCMultiGPUExchange(BMBCMultiGPUExchange &exchange)
{
    if (exchange.remote_ready)
    {
        checkCuda(cudaSetDevice(exchange.remote_device));
        checkCuda(cudaEventDestroy(exchange.remote_ready));
    }
    if (exchange.remote_consumed)
    {
        checkCuda(cudaSetDevice(exchange.owner_device));
        checkCuda(cudaEventDestroy(exchange.remote_consumed));
    }
    if (exchange.d_remote)
    {
        checkCuda(cudaSetDevice(exchange.owner_device));
        checkCuda(cudaFree(exchange.d_remote));
    }
    if (exchange.h_remote)
        checkCuda(cudaFreeHost(exchange.h_remote));
    exchange = BMBCMultiGPUExchange{};
}

void allocateBMBCProfiles(BMBCProfiles &monitor,
                          const std::size_t block_samples,
                          const std::size_t capacity_blocks)
{
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

    printPoint(BMBC_UX_Y_OFFSET + NY - 2, "ux(y=NY-2)");
    printPoint(BMBC_UY_X_OFFSET + NX - 2, "uy(x=NX-2)");
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
