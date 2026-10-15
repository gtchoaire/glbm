
#include "../include/globalFunctions.cuh"
#include "../include/output.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <ctime>

#include PROPERTIES

namespace
{
constexpr std::array<const char *, 4> FIELD_NAMES{"rho", "ux", "uy", "uz"};
constexpr const char *PRECISION_NAME = sizeof(dfloat) == sizeof(float) ? "float" : "double";
constexpr const char *VTK_SCALAR_TYPE = sizeof(dfloat) == sizeof(float) ? "Float32" : "Float64";
}

bool createOutputDirectory(std::filesystem::path &output_directory)
{
    const auto now = std::time(nullptr);
    std::tm local_time{};
    if (localtime_r(&now, &local_time) == nullptr)
        return false;

    std::ostringstream directory_name;
    directory_name << std::put_time(&local_time, "%Y%m%d_%H%M");
    output_directory = std::filesystem::path("output") / directory_name.str();

    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    return !error;
}

bool writeSimulationInfo(const std::filesystem::path &output_directory)
{
    const std::filesystem::path path = output_directory / "info.txt";
    std::ofstream file(path);
    if (!file) return false;

    file << std::setprecision(std::numeric_limits<dfloat>::max_digits10)
         << "Re = " << RE << '\n'
         << "Nx = " << GLOBAL_NX << '\n'
         << "Ny = " << GLOBAL_NY << '\n'
         << "Nz = " << GLOBAL_NZ << '\n'
         << "SAR = " << SAR << '\n'
         << "Umax = " << U_MAX << '\n'
         << "Stencil = " << STR(STENCIL) << '\n'
         << "Precision = " << PRECISION_NAME << '\n'
         << "Nsteps = " << N_STEPS << '\n'
         << "Kinetic energy output = " << (SAVE_KINETIC_ENERGY_OUTPUT ? "true" : "false") << '\n'
         << "Kinetic energy interval = " << KINETIC_ENERGY_INTERVAL << '\n'
         << "Checkpoint file = " << (output_directory / std::filesystem::path(CHECKPOINT_PATH)) << '\n'
         << "block x = " << BLOCK_NX << '\n'
         << "block y = " << BLOCK_NY << '\n'
         << "block z = " << BLOCK_NZ << '\n';

    return static_cast<bool>(file);
}

bool writeVti(const std::vector<LBMState *> &states, const int step,
              const std::filesystem::path &output_directory)
{
    if (states.size() != static_cast<size_t>(NUM_GPUS)) return false;

    std::ostringstream filename;
    filename << "state_" << std::setw(8) << std::setfill('0')
             << step << ".vti";
    const std::filesystem::path path = output_directory / filename.str();
    std::ofstream file(path);
    if (!file) return false;

    file << std::setprecision(std::numeric_limits<dfloat>::max_digits10);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"ImageData\" version=\"0.1\" "
            "byte_order=\"LittleEndian\">\n"
         << "  <ImageData WholeExtent=\"0 " << GLOBAL_NX - 1
         << " 0 " << GLOBAL_NY - 1 << " 0 " << GLOBAL_NZ - 1
         << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
         << "    <Piece Extent=\"0 " << GLOBAL_NX - 1
         << " 0 " << GLOBAL_NY - 1 << " 0 " << GLOBAL_NZ - 1 << "\">\n"
         << "      <PointData Scalars=\"density\" Vectors=\"velocity\">\n"
         << "        <DataArray type=\""
         << VTK_SCALAR_TYPE
         << "\" Name=\"density\" format=\"ascii\">\n";

    for (const LBMState *state : states)
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                {
                    const size_t index = idxScalarBlock(
                        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                        x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                    file << RHO_0 + state->host(M_RHO_INDEX, index) << ' ';
                }

    file << "\n        </DataArray>\n"
         << "        <DataArray type=\""
         << VTK_SCALAR_TYPE
         << "\" Name=\"velocity\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";

    for (const LBMState *state : states)
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                {
                    const size_t index = idxScalarBlock(
                        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                        x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                    file << state->host(M_UX_INDEX, index) / F_M_I_SCALE << ' '
                         << state->host(M_UY_INDEX, index) / F_M_I_SCALE << ' '
                         << state->host(M_UZ_INDEX, index) / F_M_I_SCALE << ' ';
                }

    file << "\n        </DataArray>\n"
         << "      </PointData>\n"
         << "      <CellData/>\n"
         << "    </Piece>\n"
         << "  </ImageData>\n"
         << "</VTKFile>\n";

    return static_cast<bool>(file);
}

bool writeBinaryFields(const std::vector<LBMState *> &states, const int step,
                       const std::filesystem::path &output_directory)
{
    std::ostringstream suffix;
    suffix << std::setw(8) << std::setfill('0') << step;

    std::array<std::ofstream, 4> files;
    for (size_t field = 0; field < FIELD_NAMES.size(); ++field)
    {
        const std::filesystem::path path = output_directory / (std::string(FIELD_NAMES[field]) + "_" + suffix.str() + ".bin");
        files[field].open(path, std::ios::binary);
        if (!files[field]) return false;
    }

    // Consecutive dfloat values in global z-y-x order, without a header.
    for (const LBMState *state : states)
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                {
                    const size_t index = idxScalarBlock(
                        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                        x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);

                    const dfloat values[4]{
                        RHO_0 + state->host(M_RHO_INDEX, index),
                        state->host(M_UX_INDEX, index) / F_M_I_SCALE,
                        state->host(M_UY_INDEX, index) / F_M_I_SCALE,
                        state->host(M_UZ_INDEX, index) / F_M_I_SCALE};

                    for (size_t field = 0; field < FIELD_NAMES.size(); ++field)
                        files[field].write(reinterpret_cast<const char *>(&values[field]), sizeof(values[field]));
                }

    for (const std::ofstream &file : files)
        if (!file) return false;
    return true;
}
