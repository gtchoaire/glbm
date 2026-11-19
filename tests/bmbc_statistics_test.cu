#include "bmbc.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{
bool close(const double actual, const double expected)
{
    const double scale = std::max(1.0, std::abs(expected));
    return std::abs(actual - expected) <= 1.0e-12 * scale;
}
}

int main()
{
    BMBCProfiles monitor;
    allocateBMBCProfiles(monitor, 1, 4);

    const double values[] = {1.0, 2.0, 4.0, 8.0};
    std::vector<double> batch_means(4 * BMBC_PROFILE_POINTS);
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t p = 0; p < BMBC_PROFILE_POINTS; ++p)
            batch_means[k * BMBC_PROFILE_POINTS + p] = values[k];

    cudaError_t status = cudaMemcpy(
        monitor.d_batch_means, batch_means.data(),
        batch_means.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    monitor.completed_blocks = 4;
    monitor.total_samples = 4;

    computeBMBCStatistics(monitor, nullptr);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cerr << "BMBC kernel failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    std::vector<double> statistics;
    copyBMBCStatisticsToHost(monitor, statistics);
    const double expected_mean = 3.75;
    const double expected_s0 = 28.75;
    // sum(X_k X_{k+1} - mean^2), deliberately different from the
    // conventional centered lag-one covariance for this finite series.
    const double expected_s1 = -0.1875;
    const double expected_variance = 28.375 / 6.0;
    const std::size_t stride = BMBC_PROFILE_POINTS;

    const bool ok =
        close(statistics[BMBC_STAT_MEAN * stride], expected_mean) &&
        close(statistics[BMBC_STAT_S0 * stride], expected_s0) &&
        close(statistics[BMBC_STAT_S1 * stride], expected_s1) &&
        close(statistics[BMBC_STAT_VARIANCE * stride], expected_variance);

    if (!ok)
    {
        std::cerr << "Unexpected BMBC result: mean="
                  << statistics[BMBC_STAT_MEAN * stride]
                  << " S0=" << statistics[BMBC_STAT_S0 * stride]
                  << " S1=" << statistics[BMBC_STAT_S1 * stride]
                  << " variance=" << statistics[BMBC_STAT_VARIANCE * stride]
                  << '\n';
    }

    freeBMBCProfiles(monitor);
    return ok ? 0 : 1;
}
