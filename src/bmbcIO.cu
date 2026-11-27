#include "../include/bmbcIO.cuh"
#include "../include/globalFunctions.cuh"
#include "../include/definitions.h"

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
constexpr int BMBC_IO_VERSION = 4;

std::filesystem::path snapshotPath(const std::filesystem::path &statistics_directory)
{
    return statistics_directory / "bmbc_blocks.bin";
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

} // namespace

std::filesystem::path writeBMBCSnapshot(
    const BMBCProfiles &monitor,
    const std::size_t completed_step,
    const std::size_t start_step,
    const std::size_t sample_interval,
    const std::filesystem::path &statistics_directory)
{
    require(monitor.d_batch_sum && monitor.d_batch_means,
            "monitor is not allocated");
    validateCounters(monitor.block_samples, monitor.completed_blocks,
                     monitor.samples_in_block, monitor.total_samples);

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

    const std::filesystem::path target = snapshotPath(statistics_directory);
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
        << std::setprecision(std::numeric_limits<dfloat>::max_digits10)
        << "reynolds " << RE << '\n'
        << "u_max " << U_MAX << '\n'
        << "aspect_ratio " << SAR << '\n'
        << "step " << completed_step << '\n'
        << "start_step " << start_step << '\n'
        << "sample_interval " << sample_interval << '\n'
        << "block_samples " << monitor.block_samples << '\n'
        << "completed_blocks " << monitor.completed_blocks << '\n'
        << "samples_in_block " << monitor.samples_in_block << '\n'
        << "total_samples " << monitor.total_samples << '\n'
        << "fields " << NUM_LOCAL_FIELDS << '\n'
        << "profile_points " << BMBC_PROFILE_POINTS << '\n'
        << "data\n";
    require(static_cast<bool>(out), "failed to write metadata");
    writeBytes(out, partial_sum);
    writeBytes(out, history);
    out.close();
    require(static_cast<bool>(out), "failed to finish snapshot");
    return target;
}

void restoreBMBCSnapshot(
    BMBCProfiles &monitor,
    const std::size_t expected_step,
    const std::size_t expected_samples,
    const std::size_t expected_start_step,
    const std::size_t expected_sample_interval,
    const std::filesystem::path &statistics_directory)
{
    require(monitor.d_batch_sum && monitor.d_batch_means,
            "allocateBMBCProfiles must be called before restore");
    require(monitor.total_samples == 0 && monitor.completed_blocks == 0 &&
                monitor.samples_in_block == 0,
            "cannot restore into nonempty BMBC state");

    const std::filesystem::path source = snapshotPath(statistics_directory);
    std::ifstream in(source, std::ios::binary);
    require(static_cast<bool>(in), "missing snapshot " + source.string());

    int format_version = 0;
    std::size_t nx = 0, ny = 0, nz = 0;
    std::size_t step = 0, block_samples = 0, completed = 0, partial = 0, total = 0;
    std::size_t start_step = 0, sample_interval = 0;
    std::size_t fields = 0, profile_points = 0;
    dfloat reynolds = 0, u_max = 0;
    double aspect_ratio = 0.0;
    readEntry(in, "GLBM_BMBC", format_version);
    require(format_version == BMBC_IO_VERSION, "unsupported snapshot version");
    readEntry(in, "nx", nx);
    readEntry(in, "ny", ny);
    readEntry(in, "nz", nz);
    readEntry(in, "reynolds", reynolds);
    readEntry(in, "u_max", u_max);
    readEntry(in, "aspect_ratio", aspect_ratio);
    readEntry(in, "step", step);
    readEntry(in, "start_step", start_step);
    readEntry(in, "sample_interval", sample_interval);
    readEntry(in, "block_samples", block_samples);
    readEntry(in, "completed_blocks", completed);
    readEntry(in, "samples_in_block", partial);
    readEntry(in, "total_samples", total);
    readEntry(in, "fields", fields);
    readEntry(in, "profile_points", profile_points);
    std::string data_line;
    require(static_cast<bool>(std::getline(in, data_line)) && data_line == "data",
            "missing data marker");

    require(fields == NUM_LOCAL_FIELDS && profile_points == BMBC_PROFILE_POINTS,
            "profile layout mismatch");
    require(nx == GLOBAL_NX && ny == GLOBAL_NY && nz == GLOBAL_NZ,
            "domain mismatch");
    require(reynolds == RE && u_max == U_MAX &&
                aspect_ratio == static_cast<double>(SAR),
            "physical parameters mismatch");
    require(start_step == expected_start_step &&
                sample_interval == expected_sample_interval,
            "sampling window mismatch");
    require(step == expected_step, "checkpoint step mismatch");
    require(block_samples == monitor.block_samples, "block_samples mismatch");
    require(total == expected_samples, "BMBC sample count mismatch");
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
