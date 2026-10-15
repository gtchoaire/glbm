#include "../include/checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../include/definitions.h"
#include "../include/globalFunctions.cuh"
#include "../include/lbmState.cuh"

namespace
{
constexpr int CHECKPOINT_VERSION = 2;
constexpr int MIN_SUPPORTED_CHECKPOINT_VERSION = 1;

constexpr const char *precisionName()
{
    return (sizeof(dfloat)) == sizeof(float) ? "float32" : "float64";
}

void require(const bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error("Checkpoint: " + message);
}

std::string readLine(std::ifstream &input)
{
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "incomplete header");
    return line;
}

template<typename... Values>
void parseLine(
    const std::string &line,
    const char *expected_key,
    Values &...values)
{
    std::istringstream parser(line);
    std::string key;
    parser >> key;
    require(key == expected_key, "expected header field '" + std::string(expected_key) + "'");
    require(static_cast<bool>((parser >> ... >> values)), "invalid value for '" + std::string(expected_key) + "'");
    parser >> std::ws;
    require(parser.eof(), "extra value in '" + std::string(expected_key) + "'");
}

constexpr size_t scalarIndex(const int x, const int y, const int z)
{
    return idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
}
}

void writeCheckpoint(
    const std::vector<LBMState *> &states,
    const size_t completed_step,
    const std::string &path_string,
    const std::filesystem::path &output_directory)
{
    require(states.size() == static_cast<size_t>(NUM_GPUS), "incorrect number of subdomains while writing");

    const std::filesystem::path path(path_string);
    require(!path.empty(), "empty output path");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());

    std::filesystem::path temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot open temporary output file");

    output << "GLBM_CHECKPOINT " << CHECKPOINT_VERSION << '\n'
           << "precision " << precisionName() << '\n'
           << "stencil " << STR(STENCIL) << '\n'
           << "dimensions " << GLOBAL_NX << ' ' << GLOBAL_NY << ' '
           << GLOBAL_NZ << '\n'
           << "moments " << NUMBER_MOMENTS << '\n'
           << "reconstruction_order " << ORDER << '\n'
           << std::setprecision(std::numeric_limits<dfloat>::max_digits10)
           << "reynolds " << RE << '\n'
           << "u_max " << U_MAX << '\n'
           << "output_directory "
           << std::quoted(std::filesystem::absolute(output_directory)
                              .lexically_normal().string()) << '\n'
           << "state post_collision_moments\n"
           << "step " << completed_step << '\n'
           << "data\n";

    for (const LBMState *state : states)
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                {
                    const size_t index = scalarIndex(x, y, z);
                    for (int moment = 0; moment < NUMBER_MOMENTS; ++moment)
                    {
                        const dfloat value = state->host(moment, index);
                        output.write(reinterpret_cast<const char *>(&value), sizeof(value));
                    }
                }

    output.close();
    require(static_cast<bool>(output), "failed while writing data");

    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    require(!error, "cannot publish completed file: " + error.message());
}

size_t loadCheckpoint(
    const std::vector<std::unique_ptr<DeviceDomain>> &domains,
    const std::vector<LBMState *> &states,
    const std::string &path,
    std::filesystem::path &output_directory)
{
    require(domains.size() == states.size(), "inconsistent destination domains");
    require(states.size() == static_cast<size_t>(NUM_GPUS), "incorrect number of destination subdomains");

    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open restart file '" + path + "'");

    std::string magic;
    int version = 0;
    parseLine(readLine(input), "GLBM_CHECKPOINT", version);
    require(version >= MIN_SUPPORTED_CHECKPOINT_VERSION && version <= CHECKPOINT_VERSION, "unsupported format version");

    std::string precision;
    parseLine(readLine(input), "precision", precision);
    require(precision == precisionName(), "precision does not match this build");

    std::string stencil;
    parseLine(readLine(input), "stencil", stencil);
    require(stencil == STR(STENCIL), "stencil does not match this build");

    int nx = 0;
    int ny = 0;
    int nz = 0;
    parseLine(readLine(input), "dimensions", nx, ny, nz);
    require(nx == GLOBAL_NX && ny == GLOBAL_NY && nz == GLOBAL_NZ, "domain dimensions do not match this build");

    int moments = 0;
    parseLine(readLine(input), "moments", moments);
    require(moments == static_cast<int>(NUMBER_MOMENTS), "moment count does not match this build");

    int reconstruction_order = 0;
    parseLine(readLine(input), "reconstruction_order", reconstruction_order);
    require(reconstruction_order == ORDER, "reconstruction order does not match this build");

    dfloat reynolds = 0;
    parseLine(readLine(input), "reynolds", reynolds);
    require(reynolds == RE, "Reynolds number does not match this build");

    dfloat u_max = 0;
    parseLine(readLine(input), "u_max", u_max);
    require(u_max == U_MAX, "maximum velocity does not match this build");

    output_directory.clear();
    if (version >= 2)
    {
        const std::string line = readLine(input);
        std::istringstream parser(line);
        std::string key;
        std::string stored_output_directory;
        parser >> key >> std::quoted(stored_output_directory);
        require(key == "output_directory", "expected header field 'output_directory'");
        require(!stored_output_directory.empty(), "empty output directory");
        parser >> std::ws;
        require(parser.eof(), "extra value in 'output_directory'");
        output_directory = stored_output_directory;
    }

    std::string state_representation;
    parseLine(readLine(input), "state", state_representation);
    require(state_representation == "post_collision_moments", "unsupported state representation");

    size_t completed_step = 0;
    parseLine(readLine(input), "step", completed_step);
    require(readLine(input) == "data", "missing data marker");

    for (LBMState *state : states)
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                {
                    const size_t index = scalarIndex(x, y, z);
                    for (int moment = 0; moment < NUMBER_MOMENTS; ++moment)
                    {
                        dfloat value = 0;
                        input.read(reinterpret_cast<char *>(&value), sizeof(value));
                        require(static_cast<bool>(input), "truncated moment data");
                        state->h_moments[LBMState::storageIndex(moment, index)] = value;
                    }
                }

    char trailing_byte = 0;
    input.read(&trailing_byte, 1);
    require(input.gcount() == 0, "unexpected trailing data");

    for (size_t subdomain = 0; subdomain < states.size(); ++subdomain)
    {
        checkCuda(cudaSetDevice(domains[subdomain]->device_id));
        checkCuda(cudaMemcpy(states[subdomain]->d_moments,
            states[subdomain]->h_moments, MEM_SIZE_MOM,
            cudaMemcpyHostToDevice));
    }

    return completed_step;
}
