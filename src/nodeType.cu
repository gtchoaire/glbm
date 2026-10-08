#include "../include/nodeType.h"

#include <iostream>
#include <new>
#include <cuda_runtime.h>

#include "../include/globalFunctions.cuh"

namespace
{
unsigned int nodeTypeAt(
    const int x, const int y, const int local_z,
    const SubDomainInfo &subdomain)
{
    // The complete north face is the moving lid, including edges and corners.
    if (y == NY - 1)
        return NORTH;

    const int global_z = subdomain.global_z_start + local_z;
    unsigned int type = BULK;

    if (y == 0) type |= SOUTH;
    if (x == 0) type |= WEST;
    if (x == NX - 1) type |= EAST;
    if (subdomain.has_back_wall && global_z == 0) type |= BACK;
    if (subdomain.has_front_wall && global_z == subdomain.global_nz - 1)
        type |= FRONT;

    return type;
}

bool checkCuda(const cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::cerr << "CUDA error while " << operation << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}
}

bool initializeNodeTypes(DeviceDomain &domain)
{
    domain.h_node_type = new (std::nothrow) unsigned int[NUMBER_LBM_NODES];
    if (domain.h_node_type == nullptr)
        return false;

    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
            {
                const size_t index = idxScalarBlock(
                    x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                    x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                domain.h_node_type[index] = nodeTypeAt(x, y, z, domain.info);
            }

    const size_t bytes = NUMBER_LBM_NODES * sizeof(unsigned int);
    if (!checkCuda(cudaMalloc(
            reinterpret_cast<void **>(&domain.d_node_type), bytes),
            "allocating the node map") ||
        !checkCuda(cudaMemcpy(
            domain.d_node_type, domain.h_node_type, bytes,
            cudaMemcpyHostToDevice),
            "copying the node map"))
    {
        freeNodeTypes(domain);
        return false;
    }
    return true;
}

void freeNodeTypes(DeviceDomain &domain)
{
    delete[] domain.h_node_type;
    domain.h_node_type = nullptr;
    if (domain.d_node_type != nullptr)
        cudaFree(domain.d_node_type);
    domain.d_node_type = nullptr;
}
