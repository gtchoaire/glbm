#ifndef LOCAL_FIELD_STATISTICS_IO_H
#define LOCAL_FIELD_STATISTICS_IO_H

#include <cstddef>
#include <filesystem>
#include <vector>

struct LocalFieldStatisticsSnapshot
{
    std::size_t step = 0;
    std::size_t start_step = 0;
    std::size_t end_step = 0;
    std::size_t sample_interval = 1;
    std::size_t samples = 0;
    std::vector<std::vector<double>> means;
};

std::filesystem::path localFieldStatisticsPath(
    const std::filesystem::path &statistics_directory);

std::filesystem::path writeLocalFieldStatistics(
    const LocalFieldStatisticsSnapshot &snapshot,
    const std::filesystem::path &statistics_directory);

LocalFieldStatisticsSnapshot readLocalFieldStatistics(
    const std::filesystem::path &statistics_directory,
    std::size_t expected_step);

#endif
