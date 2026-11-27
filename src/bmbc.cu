#include "../include/bmbc.cuh"
#include "../include/globalFunctions.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace
{
    constexpr std::size_t BMBC_BLOCK_MULTIPLE = 60;
    constexpr double BMBC_CORRELATION_LIMIT = 0.5;

    __global__ void gpuAccumulateBMBCProfiles(
        const LBMState state, const int global_z_start, double *sum)
    {
        std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= BMBC_LINE_NODES)
            return;

        int line = 0;
        while (index >= bmbcLineSize(line))
            index -= bmbcLineSize(line++);

        const bool along_x = line == BMBC_X_CENTER;
        double values[NUM_LOCAL_FIELDS];

        for (int corner = 0; corner < 4; ++corner)
        {
            const int first = corner & 1;
            const int second = corner >> 1;
            const int x = along_x ? static_cast<int>(index) : (GLOBAL_NX - 1 + first) / 2;
            const int y = along_x ? (GLOBAL_NY - 1 + first) / 2 : static_cast<int>(index);
            const int z = (GLOBAL_NZ - 1 + second) / 2;
            const int local_z = z - global_z_start;
            if (local_z < 0 || local_z >= NZ)
                continue;

            evaluateLocalFields(
                state,
                idxScalarBlock(
                    x % BLOCK_NX, y % BLOCK_NY, local_z % BLOCK_NZ,
                    x / BLOCK_NX, y / BLOCK_NY, local_z / BLOCK_NZ),
                values);
            for (int field = 0; field < NUM_LOCAL_FIELDS; ++field)
                sum[bmbcPoint(line, field, index)] += 0.25 * values[field];
        }
    }

    void launchBMBCAccumulation(
        const LBMState &state, const SubDomainInfo &info,
        double *sum, const cudaStream_t stream)
    {
        constexpr int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            (BMBC_LINE_NODES + threads - 1) / threads);
        gpuAccumulateBMBCProfiles<<<blocks, threads, 0, stream>>>(
            state, info.global_z_start, sum);
        checkCuda(cudaGetLastError());
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

    launchBMBCAccumulation(state, info, local.d_sum, stream);
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
    if (monitor.d_batch_sum || monitor.d_batch_means)
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

    monitor.block_samples = block_samples;
    monitor.capacity_blocks = capacity_blocks;
    monitor.samples_in_block = 0;
    monitor.completed_blocks = 0;
    monitor.total_samples = 0;
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
                            const SubDomainInfo &info,
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

    launchBMBCAccumulation(state, info, monitor.d_batch_sum, stream);

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

std::size_t usableBMBCBlocks(const std::size_t count)
{
    return count < BMBC_BLOCK_MULTIPLE ? count : count - count % BMBC_BLOCK_MULTIPLE;
}

BMBCEstimate estimateBMBC(const std::vector<double> &series)
{
    const std::size_t total = series.size();
    BMBCEstimate estimate;
    estimate.mean = std::accumulate(series.begin(), series.end(), 0.0) / static_cast<double>(total);
    estimate.error = std::numeric_limits<double>::quiet_NaN();

    for (std::size_t size = 1; size <= total / 3; ++size)
    {
        if (total % size != 0)
            continue;

        const std::size_t groups = total / size;
        double s0 = 0.0;
        double s1 = 0.0;
        double previous = 0.0;

        for (std::size_t group = 0; group < groups; ++group)
        {
            double sum = 0.0;
            for (std::size_t k = group * size; k < (group + 1) * size; ++k)
                sum += series[k] - estimate.mean;

            const double centered = sum / static_cast<double>(size);
            s0 += centered * centered;
            s1 += previous * centered;
            previous = centered;
        }

        if (s1 <= BMBC_CORRELATION_LIMIT * s0)
        {
            estimate.error = std::sqrt(
                (s0 + 2.0 * s1) / (static_cast<double>(groups - 1) * static_cast<double>(groups - 2)));
            estimate.group = size;
            estimate.groups = groups;
            break;
        }
    }

    return estimate;
}

void reportBMBCProfiles(
    const BMBCProfiles &monitor,
    const std::size_t completed_steps,
    const cudaStream_t stream)
{
    const std::size_t blocks = usableBMBCBlocks(monitor.completed_blocks);
    if (blocks < 3)
        return;

    checkCuda(cudaStreamSynchronize(stream));
    std::vector<double> means;
    copyBMBCCompletedMeansToHost(monitor, means);

    struct Row
    {
        int line, first, second, product;
    };

    constexpr Row rows[] = {
        {BMBC_Y_CENTER, LOCAL_FIELD_UX, LOCAL_FIELD_UX, -1},
        {BMBC_X_CENTER, LOCAL_FIELD_UY, LOCAL_FIELD_UY, -1},
        {BMBC_Y_CENTER, LOCAL_FIELD_UX, LOCAL_FIELD_UX, LOCAL_FIELD_UX2},
        {BMBC_Y_CENTER, LOCAL_FIELD_UX, LOCAL_FIELD_UY, LOCAL_FIELD_UXUY},
        {BMBC_X_CENTER, LOCAL_FIELD_UY, LOCAL_FIELD_UY, LOCAL_FIELD_UY2},
        {BMBC_X_CENTER, LOCAL_FIELD_UX, LOCAL_FIELD_UY, LOCAL_FIELD_UXUY}};

    double velocity_error = 0.0;
    double stress_error = 0.0;
    std::vector<double> series(blocks), first(blocks), second(blocks);

    for (const Row &row : rows)
    {
        const std::size_t size = bmbcLineSize(row.line);
        const auto extract = [&](std::vector<double> &target, const int field, const std::size_t index)
        {
            for (std::size_t k = 0; k < blocks; ++k)
                target[k] = means[k * BMBC_PROFILE_POINTS + bmbcPoint(row.line, field, index)];
            return std::accumulate(target.begin(), target.end(), 0.0) / static_cast<double>(blocks);
        };

        double peak = 0.0;
        double error = 0.0;

        for (std::size_t index = 1; index + 1 < size; ++index)
        {
            const double a = extract(first, row.first, index);
            const double b = extract(second, row.second, index);

            if (row.product < 0)
                series = first;
            else
            {
                extract(series, row.product, index);
                for (std::size_t k = 0; k < blocks; ++k)
                    series[k] += a * b - a * second[k] - b * first[k];
            }

            const BMBCEstimate estimate = estimateBMBC(series);
            peak = std::max(peak, std::abs(estimate.mean));

            error = std::max(
                error, std::isfinite(estimate.error) ? estimate.error : std::numeric_limits<double>::infinity());
        }

        double &largest = row.product < 0 ? velocity_error : stress_error;
        if (peak > 0.0)
            largest = std::max(largest, 100.0 * error / peak);
    }

    std::printf(
        "\nBMBC | t* %.1f | Max error: velocity %.1f%% | Reynolds stress %.1f%%\n",
        static_cast<double>(completed_steps) * static_cast<double>(U_MAX) /
            static_cast<double>(GLOBAL_NX - 1),
        velocity_error, stress_error);
}

void freeBMBCProfiles(BMBCProfiles &monitor)
{
    if (monitor.d_batch_sum)
        checkCuda(cudaFree(monitor.d_batch_sum));
    if (monitor.d_batch_means)
        checkCuda(cudaFree(monitor.d_batch_means));
    monitor = BMBCProfiles{};
}
