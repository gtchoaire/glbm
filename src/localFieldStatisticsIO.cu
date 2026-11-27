#include "../include/localFieldStatisticsIO.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../include/definitions.h"
#include "../include/globalFunctions.cuh"
#include "../include/localFieldStatistics.cuh"

namespace
{
constexpr int FORMAT_VERSION = 2;
constexpr std::array<const char *, NUM_LOCAL_FIELDS> FIELD_NAMES{
    "ux_mean", "uy_mean", "uz_mean",
    "Sxx_mean", "Sxy_mean", "Sxz_mean", "Syy_mean", "Syz_mean", "Szz_mean",
    "ux2_mean", "uy2_mean", "uz2_mean", "uxuy_mean", "uxuz_mean", "uyuz_mean",
    "S2_mean"};

void require(const bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error("Local field statistics: " + message);
}

std::string readLine(std::ifstream &input)
{
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "incomplete header");
    return line;
}

template<typename... Values>
void parseLine(const std::string &line, const char *expected_key, Values &...values)
{
    std::istringstream parser(line);
    std::string key;
    parser >> key;
    require(key == expected_key, "expected header field '" + std::string(expected_key) + "'");
    require(static_cast<bool>((parser >> ... >> values)), "invalid value for '" + std::string(expected_key) + "'");
    parser >> std::ws;
    require(parser.eof(), "extra value in '" + std::string(expected_key) + "'");
}

constexpr std::size_t scalarIndex(const int x, const int y, const int z)
{
    return idxScalarBlock(
        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
        x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
}
}

std::filesystem::path localFieldStatisticsPath(
    const std::filesystem::path &statistics_directory)
{
    return statistics_directory / "local_fields.bin";
}

std::filesystem::path writeLocalFieldStatistics(
    const LocalFieldStatisticsSnapshot &snapshot,
    const std::filesystem::path &statistics_directory)
{
    require(snapshot.sample_interval > 0, "sample interval must be positive");
    require(snapshot.end_step > snapshot.start_step, "invalid sampling window");
    require(snapshot.samples > 0, "cannot write zero samples");
    require(snapshot.step > snapshot.start_step, "statistics precede the sampling window");
    const std::size_t last_step = std::min(snapshot.step, snapshot.end_step);
    require(snapshot.samples ==
                (last_step - snapshot.start_step) / snapshot.sample_interval,
            "sample count does not match the simulation step");
    require(snapshot.means.size() == static_cast<std::size_t>(NUM_GPUS),
            "incorrect number of subdomains");
    for (const auto &local_means : snapshot.means)
        require(local_means.size() == static_cast<std::size_t>(NUM_LOCAL_FIELDS) * NUMBER_LBM_NODES,
                "invalid field buffer size");

    std::filesystem::create_directories(statistics_directory);
    const std::filesystem::path target = localFieldStatisticsPath(statistics_directory);
    const std::filesystem::path temporary(target.string() + ".tmp");
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot open temporary output file");

    output << "GLBM_LOCAL_FIELD_STATISTICS " << FORMAT_VERSION << '\n'
           << "dimensions " << GLOBAL_NX << ' ' << GLOBAL_NY << ' ' << GLOBAL_NZ << '\n'
           << std::setprecision(std::numeric_limits<dfloat>::max_digits10)
           << "reynolds " << RE << '\n'
           << "u_max " << U_MAX << '\n'
           << "aspect_ratio " << SAR << '\n'
           << "step " << snapshot.step << '\n'
           << "start_step " << snapshot.start_step << '\n'
           << "end_step " << snapshot.end_step << '\n'
           << "sample_interval " << snapshot.sample_interval << '\n'
           << "samples " << snapshot.samples << '\n'
           << "fields " << NUM_LOCAL_FIELDS << '\n'
           << "field_names";
    for (const char *name : FIELD_NAMES)
        output << ' ' << name;
    output << "\nprecision float64\nlayout field_major_zyx\ndata\n";

    for (int field = 0; field < NUM_LOCAL_FIELDS; ++field)
        for (const auto &local_means : snapshot.means)
            for (int z = 0; z < NZ; ++z)
                for (int y = 0; y < NY; ++y)
                    for (int x = 0; x < NX; ++x)
                    {
                        const std::size_t index =
                            static_cast<std::size_t>(field) * NUMBER_LBM_NODES +
                            scalarIndex(x, y, z);
                        const double value = local_means[index];
                        output.write(reinterpret_cast<const char *>(&value), sizeof(value));
                    }

    output.close();
    require(static_cast<bool>(output), "failed while writing data");
    return target;
}

LocalFieldStatisticsSnapshot readLocalFieldStatistics(
    const std::filesystem::path &statistics_directory,
    const std::size_t expected_step)
{
    const std::filesystem::path source = localFieldStatisticsPath(statistics_directory);
    std::ifstream input(source, std::ios::binary);
    require(static_cast<bool>(input), "cannot open '" + source.string() + "'");

    int version = 0;
    parseLine(readLine(input), "GLBM_LOCAL_FIELD_STATISTICS", version);
    require(version == FORMAT_VERSION, "unsupported format version");

    int nx = 0, ny = 0, nz = 0;
    parseLine(readLine(input), "dimensions", nx, ny, nz);
    require(nx == GLOBAL_NX && ny == GLOBAL_NY && nz == GLOBAL_NZ,
            "domain dimensions do not match this build");
    dfloat reynolds = 0;
    dfloat u_max = 0;
    double aspect_ratio = 0.0;
    parseLine(readLine(input), "reynolds", reynolds);
    parseLine(readLine(input), "u_max", u_max);
    parseLine(readLine(input), "aspect_ratio", aspect_ratio);
    require(reynolds == RE && u_max == U_MAX && aspect_ratio == static_cast<double>(SAR),
            "physical parameters do not match this build");

    LocalFieldStatisticsSnapshot snapshot;
    parseLine(readLine(input), "step", snapshot.step);
    parseLine(readLine(input), "start_step", snapshot.start_step);
    parseLine(readLine(input), "end_step", snapshot.end_step);
    parseLine(readLine(input), "sample_interval", snapshot.sample_interval);
    parseLine(readLine(input), "samples", snapshot.samples);
    require(snapshot.step == expected_step, "step does not match the solver checkpoint");

    int fields = 0;
    parseLine(readLine(input), "fields", fields);
    require(fields == NUM_LOCAL_FIELDS, "field count does not match this build");
    std::istringstream names_parser(readLine(input));
    std::string key;
    names_parser >> key;
    require(key == "field_names", "expected header field 'field_names'");
    for (const char *expected_name : FIELD_NAMES)
    {
        std::string name;
        names_parser >> name;
        require(name == expected_name, "field layout does not match this build");
    }
    names_parser >> std::ws;
    require(names_parser.eof(), "extra field name");
    std::string precision;
    std::string layout;
    parseLine(readLine(input), "precision", precision);
    parseLine(readLine(input), "layout", layout);
    require(precision == "float64", "unsupported precision");
    require(layout == "field_major_zyx", "unsupported layout");
    require(readLine(input) == "data", "missing data marker");

    snapshot.means.resize(NUM_GPUS);
    for (auto &local_means : snapshot.means)
        local_means.assign(static_cast<std::size_t>(NUM_LOCAL_FIELDS) * NUMBER_LBM_NODES, 0.0);

    for (int field = 0; field < NUM_LOCAL_FIELDS; ++field)
        for (auto &local_means : snapshot.means)
            for (int z = 0; z < NZ; ++z)
                for (int y = 0; y < NY; ++y)
                    for (int x = 0; x < NX; ++x)
                    {
                        const std::size_t index =
                            static_cast<std::size_t>(field) * NUMBER_LBM_NODES +
                            scalarIndex(x, y, z);
                        input.read(reinterpret_cast<char *>(&local_means[index]), sizeof(double));
                        require(static_cast<bool>(input), "truncated field data");
                    }

    char trailing = 0;
    input.read(&trailing, 1);
    require(input.gcount() == 0, "unexpected trailing data");
    return snapshot;
}
