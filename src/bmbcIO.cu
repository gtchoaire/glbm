#include "../include/bmbcIO.cuh"
#include "../include/globalFunctions.cuh"
#include "../include/definitions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int BMBC_IO_VERSION = BMBC_TURBULENCE_PROFILES ? 2 : 1;

std::filesystem::path sidecarPath(const std::filesystem::path &glbm_path)
{
    return std::filesystem::path(glbm_path.string() + ".bmbc");
}

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error("BMBC I/O: " + message);
}

// Read a text metadata entry, requiring exactly one value and no trailing junk.
template <typename T>
void readEntry(std::istream &input, const char *name, T &value)
{
    std::string line;
    require(static_cast<bool>(std::getline(input, line)),
            std::string("missing field ") + name);
    std::istringstream parsed(line);
    std::string tag;
    require(static_cast<bool>(parsed >> tag >> value) && tag == name,
            std::string("invalid field ") + name);
    parsed >> std::ws;
    require(parsed.eof(), std::string("trailing data in ") + name);
}

std::size_t bytesForBlocks(std::size_t blocks)
{
    const std::size_t bytes_per_block = BMBC_PROFILE_POINTS * sizeof(double);
    require(blocks <= std::numeric_limits<std::size_t>::max() / bytes_per_block,
            "block count overflow");
    return blocks * bytes_per_block;
}

void writeBytes(std::ostream &out, const std::vector<double> &values)
{
    const auto bytes = values.size() * sizeof(double);
    if (bytes > 0)
        out.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(bytes));
    require(static_cast<bool>(out), "failed to write payload");
}

void readBytes(std::istream &in, std::vector<double> &values)
{
    const auto bytes = values.size() * sizeof(double);
    if (bytes > 0)
        in.read(reinterpret_cast<char *>(values.data()),
                static_cast<std::streamsize>(bytes));
    require(static_cast<bool>(in), "truncated payload");
}

void validateCounters(const std::size_t block_samples,
                      const std::size_t completed,
                      const std::size_t partial,
                      const std::size_t total)
{
    require(block_samples > 0, "block_samples must be positive");
    require(partial < block_samples, "invalid partial block count");
    require(completed <= (std::numeric_limits<std::size_t>::max() - partial) /
                             block_samples,
            "sample count overflow");
    require(completed * block_samples + partial == total,
            "inconsistent BMBC sample counters");
}

struct ProfilePoint
{
    const char *name;
    std::size_t index;
    std::size_t dimension;
    int normalization_power;
};

ProfilePoint describeProfilePoint(const std::size_t p)
{
    if constexpr (!BMBC_TURBULENCE_PROFILES)
    {
        if (p < static_cast<std::size_t>(NY))
            return {"ux_y", p, static_cast<std::size_t>(NY), 1};
        return {"uy_x", p - NY, static_cast<std::size_t>(NX), 1};
    }

    if (p < BMBC_UY_Y_OFFSET)
        return {"ux_y", p - BMBC_UX_Y_OFFSET, static_cast<std::size_t>(NY), 1};
    if (p < BMBC_UX2_Y_OFFSET)
        return {"uy_y", p - BMBC_UY_Y_OFFSET, static_cast<std::size_t>(NY), 1};
    if (p < BMBC_UXUY_Y_OFFSET)
        return {"ux2_y", p - BMBC_UX2_Y_OFFSET, static_cast<std::size_t>(NY), 2};
    if (p < BMBC_UX_X_OFFSET)
        return {"uxuy_y", p - BMBC_UXUY_Y_OFFSET, static_cast<std::size_t>(NY), 2};
    if (p < BMBC_UY_X_TURB_OFFSET)
        return {"ux_x", p - BMBC_UX_X_OFFSET, static_cast<std::size_t>(NX), 1};
    if (p < BMBC_UY2_X_OFFSET)
        return {"uy_x", p - BMBC_UY_X_TURB_OFFSET, static_cast<std::size_t>(NX), 1};
    if (p < BMBC_UXUY_X_OFFSET)
        return {"uy2_x", p - BMBC_UY2_X_OFFSET, static_cast<std::size_t>(NX), 2};
    return {"uxuy_x", p - BMBC_UXUY_X_OFFSET, static_cast<std::size_t>(NX), 2};
}
} // namespace

void writeBMBCSnapshot(
    const BMBCProfiles &monitor,
    const std::size_t completed_step,
    const std::filesystem::path &glbm_checkpoint_file)
{
    require(monitor.d_batch_sum && monitor.d_batch_means,
            "monitor is not allocated");
    validateCounters(monitor.block_samples, monitor.completed_blocks,
                     monitor.samples_in_block, monitor.total_samples);

    // Safe for the currently supported single-GPU/default-stream workflow.
    checkCuda(cudaDeviceSynchronize());

    const std::size_t history_bytes = bytesForBlocks(monitor.completed_blocks);
    std::vector<double> partial_sum(BMBC_PROFILE_POINTS);
    std::vector<double> history(history_bytes / sizeof(double));

    checkCuda(cudaMemcpy(partial_sum.data(), monitor.d_batch_sum,
                         bytesForBlocks(1), cudaMemcpyDeviceToHost));
    if (!history.empty())
        checkCuda(cudaMemcpy(history.data(), monitor.d_batch_means,
                             bytesForBlocks(monitor.completed_blocks),
                             cudaMemcpyDeviceToHost));

    const std::filesystem::path target = sidecarPath(glbm_checkpoint_file);
    const std::filesystem::path temporary(target.string() + ".tmp");
    if (!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());

    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(out), "cannot open " + temporary.string());

    // Store global dimensions so that a checkpoint can be resumed with a
    // different supported domain decomposition (one or two GPUs).
    out << "GLBM_BMBC " << BMBC_IO_VERSION << '\n'
        << "nx " << GLOBAL_NX << '\n'
        << "ny " << GLOBAL_NY << '\n'
        << "nz " << GLOBAL_NZ << '\n'
        << "step " << completed_step << '\n'
        << "block_samples " << monitor.block_samples << '\n'
        << "completed_blocks " << monitor.completed_blocks << '\n'
        << "samples_in_block " << monitor.samples_in_block << '\n'
        << "total_samples " << monitor.total_samples << '\n';
    if constexpr (BMBC_TURBULENCE_PROFILES)
        out << "profile_points " << BMBC_PROFILE_POINTS << '\n';
    out << "data\n";
    require(static_cast<bool>(out), "failed to write metadata");
    writeBytes(out, partial_sum);
    writeBytes(out, history);
    out.close();
    require(static_cast<bool>(out), "failed to finish snapshot");

    // The parent GLBM file is published first. A crash between publications
    // is detected by the step/sample checks in restoreBMBCSnapshot().
    std::filesystem::rename(temporary, target);
}

void restoreBMBCSnapshot(
    BMBCProfiles &monitor,
    const std::size_t expected_step,
    const std::size_t expected_samples,
    const std::filesystem::path &glbm_checkpoint_file)
{
    require(monitor.d_batch_sum && monitor.d_batch_means,
            "allocateBMBCProfiles must be called before restore");
    require(monitor.total_samples == 0 && monitor.completed_blocks == 0 &&
                monitor.samples_in_block == 0,
            "cannot restore into nonempty BMBC state");

    const std::filesystem::path source = sidecarPath(glbm_checkpoint_file);
    std::ifstream in(source, std::ios::binary);
    require(static_cast<bool>(in), "missing snapshot " + source.string());

    int format_version = 0;
    std::size_t nx = 0, ny = 0, nz = 0;
    std::size_t step = 0, block_samples = 0, completed = 0, partial = 0, total = 0;
    std::size_t profile_points = 0;
    readEntry(in, "GLBM_BMBC", format_version);
    readEntry(in, "nx", nx);
    readEntry(in, "ny", ny);
    readEntry(in, "nz", nz);
    readEntry(in, "step", step);
    readEntry(in, "block_samples", block_samples);
    readEntry(in, "completed_blocks", completed);
    readEntry(in, "samples_in_block", partial);
    readEntry(in, "total_samples", total);
    if (format_version >= 2)
        readEntry(in, "profile_points", profile_points);
    std::string data_line;
    require(static_cast<bool>(std::getline(in, data_line)) && data_line == "data",
            "missing data marker");

    require(format_version == BMBC_IO_VERSION, "unsupported snapshot version");
    if constexpr (BMBC_TURBULENCE_PROFILES)
        require(profile_points == BMBC_PROFILE_POINTS,
                "profile layout mismatch");
    // Version 1 sidecars written by the original two-GPU implementation
    // stored LOCAL_NZ. Accept that legacy metadata while new snapshots use
    // the decomposition-independent global dimensions above.
    const bool legacy_two_gpu_nz =
        GLOBAL_NZ % 2 == 0 && nz == static_cast<std::size_t>(GLOBAL_NZ / 2);
    require(nx == GLOBAL_NX && ny == GLOBAL_NY &&
                (nz == GLOBAL_NZ || legacy_two_gpu_nz),
            "domain mismatch");
    require(step == expected_step, "checkpoint step mismatch");
    require(block_samples == monitor.block_samples, "block_samples mismatch");
    require(total == expected_samples, "checkpoint statistics sample mismatch");
    validateCounters(block_samples, completed, partial, total);
    const std::size_t history_bytes = bytesForBlocks(completed);
    const std::size_t expected_payload_bytes = bytesForBlocks(1) + history_bytes;
    require(expected_payload_bytes >= history_bytes, "payload size overflow");
    const auto payload_begin = in.tellg();
    require(payload_begin != std::streampos(-1), "cannot locate payload");
    in.seekg(0, std::ios::end);
    const auto payload_end = in.tellg();
    require(payload_end >= payload_begin &&
                static_cast<std::uintmax_t>(payload_end - payload_begin) ==
                    expected_payload_bytes,
            "snapshot payload size mismatch");
    in.seekg(payload_begin);

    std::vector<double> partial_sum(BMBC_PROFILE_POINTS);
    std::vector<double> history(history_bytes / sizeof(double));
    readBytes(in, partial_sum);
    readBytes(in, history);
    require(in.peek() == std::char_traits<char>::eof(), "unexpected trailing data");

    // Grow allocation only after verifying the entire file.
    if (completed > monitor.capacity_blocks)
    {
        const std::size_t capacity = completed;
        freeBMBCProfiles(monitor);
        allocateBMBCProfiles(monitor, block_samples, capacity);
    }

    checkCuda(cudaMemcpy(monitor.d_batch_sum, partial_sum.data(),
                         bytesForBlocks(1), cudaMemcpyHostToDevice));
    if (!history.empty())
        checkCuda(cudaMemcpy(monitor.d_batch_means, history.data(),
                             bytesForBlocks(completed), cudaMemcpyHostToDevice));

    monitor.completed_blocks = completed;
    monitor.samples_in_block = partial;
    monitor.total_samples = total;
    std::printf("BMBC restored | step %zu | samples %zu | blocks %zu | partial %zu\n",
                step, total, completed, partial);
}

void exportBMBCProfilesCSV(
    BMBCProfiles &monitor,
    const std::size_t completed_step,
    const std::size_t sample_interval,
    const std::filesystem::path &output_directory)
{
    require(monitor.d_batch_means && monitor.d_statistics,
            "monitor is not allocated");

    std::filesystem::create_directories(output_directory);
    const std::filesystem::path filename = output_directory /
        ("bmbc_profiles_step_" + std::to_string(completed_step) + ".csv");
    std::ofstream out(filename);
    require(static_cast<bool>(out), "cannot open " + filename.string());

    out << "step,reynolds_number,aspect_ratio,u_max,sample_interval,group_size,M,M_steps,M_t_star,K,profile,index,position_star,normalization_power,mean,mean_over_U,"
           "S0,S1,S1_over_S0,variance,standard_error,standard_error_over_U\n";
    out << std::setprecision(17);

    // For scientific comparison, only export groupings with >=16 groups.
    // This is a practical reporting filter, not proof of reliable statistics.
    std::size_t group_size = 1;
    while (monitor.completed_blocks / group_size >= 16)
    {
        computeBMBCGroupedStatistics(monitor, group_size, nullptr);
        checkCuda(cudaDeviceSynchronize());
        std::vector<double> stats;
        copyBMBCStatisticsToHost(monitor, stats);
        const std::size_t K = monitor.completed_blocks / group_size;

        for (std::size_t p = 0; p < BMBC_PROFILE_POINTS; ++p)
        {
            const ProfilePoint profile = describeProfilePoint(p);
            const double position = profile.dimension > 1
                ? -1.0 + 2.0 * static_cast<double>(profile.index) /
                             static_cast<double>(profile.dimension - 1)
                : 0.0;

            const double mean = stats[BMBC_STAT_MEAN * BMBC_PROFILE_POINTS + p];
            const double s0 = stats[BMBC_STAT_S0 * BMBC_PROFILE_POINTS + p];
            const double s1 = stats[BMBC_STAT_S1 * BMBC_PROFILE_POINTS + p];
            const double variance = stats[BMBC_STAT_VARIANCE * BMBC_PROFILE_POINTS + p];
            const double ratio = s0 != 0.0 ? s1 / s0
                                           : std::numeric_limits<double>::quiet_NaN();
            const double standard_error = variance >= 0.0
                ? std::sqrt(variance)
                : std::numeric_limits<double>::quiet_NaN();
            const double u_lid = static_cast<double>(U_MAX);
            const double normalization = profile.normalization_power == 2
                ? u_lid * u_lid : u_lid;
            const std::size_t M = group_size * monitor.block_samples;
            const std::size_t M_steps = M * sample_interval;
            const double M_t_star =
                static_cast<double>(M_steps) * u_lid /
                static_cast<double>(GLOBAL_NX - 1);

            out << completed_step << ',' << static_cast<double>(RE) << ','
                << static_cast<double>(SAR) << ',' << u_lid << ','
                << sample_interval << ',' << group_size << ','
                << M << ',' << M_steps << ',' << M_t_star << ',' << K << ','
                << profile.name << ',' << profile.index << ','
                << position << ',' << profile.normalization_power << ','
                << mean << ',' << mean / normalization << ','
                << s0 << ',' << s1 << ',' << ratio << ',' << variance << ','
                << standard_error << ',' << standard_error / normalization << '\n';
        }

        if (group_size > monitor.completed_blocks / 2)
            break;
        group_size *= 2;
    }

    out.close();
    require(static_cast<bool>(out), "failed to write CSV");
    std::printf("BMBC full profiles exported: %s\n", filename.string().c_str());
}
