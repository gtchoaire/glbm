#include "../include/output.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "../include/globalFunctions.cuh"

#include PROPERTIES

bool writeVti(const LBMState &state, const int step)
{
    const std::filesystem::path output_directory("output");
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    if (error)
    {
        std::cerr << "Failed to create output directory: "
                  << error.message() << '\n';
        return false;
    }

    std::ostringstream filename;
    filename << "state_" << std::setw(8) << std::setfill('0')
             << step << ".vti";
    const std::filesystem::path path = output_directory / filename.str();
    std::ofstream file(path);
    if (!file)
    {
        std::cerr << "Failed to open " << path << " for writing.\n";
        return false;
    }

    file << std::setprecision(sizeof(dfloat) == sizeof(float) ? 9 : 17);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"ImageData\" version=\"0.1\" "
            "byte_order=\"LittleEndian\">\n"
         << "  <ImageData WholeExtent=\"0 " << NX - 1
         << " 0 " << NY - 1 << " 0 " << NZ - 1
         << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
         << "    <Piece Extent=\"0 " << NX - 1
         << " 0 " << NY - 1 << " 0 " << NZ - 1 << "\">\n"
         << "      <PointData Scalars=\"density\" Vectors=\"velocity\">\n"
         << "        <DataArray type=\""
         << (sizeof(dfloat) == sizeof(float) ? "Float32" : "Float64")
         << "\" Name=\"density\" format=\"ascii\">\n";

    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
            {
                const size_t index = idxScalarBlock(
                    x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                    x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                file << RHO_0 + state.host(M_RHO_INDEX, index) << ' ';
            }

    file << "\n        </DataArray>\n"
         << "        <DataArray type=\""
         << (sizeof(dfloat) == sizeof(float) ? "Float32" : "Float64")
         << "\" Name=\"velocity\" NumberOfComponents=\"3\" "
            "format=\"ascii\">\n";

    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
            {
                const size_t index = idxScalarBlock(
                    x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                    x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                file << state.host(M_UX_INDEX, index) / F_M_I_SCALE << ' '
                     << state.host(M_UY_INDEX, index) / F_M_I_SCALE << ' '
                     << state.host(M_UZ_INDEX, index) / F_M_I_SCALE << ' ';
            }

    file << "\n        </DataArray>\n"
         << "      </PointData>\n"
         << "      <CellData/>\n"
         << "    </Piece>\n"
         << "  </ImageData>\n"
         << "</VTKFile>\n";

    if (!file)
    {
        std::cerr << "Failed while writing " << path << ".\n";
        return false;
    }

    std::cout << "Wrote " << path << '\n';
    return true;
}
