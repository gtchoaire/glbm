#ifndef GLOBAL_STRUCTS_H
#define GLOBAL_STRUCTS_H

#include "var.h"

struct SubDomainInfo
{
    int gpu_id;
    int global_z_start;
    int global_z_end;
    int local_nz;
    int global_nz;
    bool has_back_wall;
    bool has_front_wall;

    __host__ __device__
    SubDomainInfo(
        int gpu,
        int z_start,
        int z_end,
        int nz_local,
        int nz_global,
        bool back_wall,
        bool front_wall) : gpu_id(gpu),
                           global_z_start(z_start),
                           global_z_end(z_end),
                           local_nz(nz_local),
                           global_nz(nz_global),
                           has_back_wall(back_wall),
                           has_front_wall(front_wall)
    {
    }
};

struct LBMState
{
    // Host pointers for LBM state variables
    dfloat *h_rho = nullptr;
    dfloat *h_ux = nullptr;
    dfloat *h_uy = nullptr;
    dfloat *h_uz = nullptr;
    dfloat *h_mxx = nullptr;
    dfloat *h_mxy = nullptr;
    dfloat *h_mxz = nullptr;
    dfloat *h_myy = nullptr;
    dfloat *h_myz = nullptr;
    dfloat *h_mzz = nullptr;

    // Device pointers for LBM state variables
    dfloat *d_rho = nullptr;
    dfloat *d_ux = nullptr;
    dfloat *d_uy = nullptr;
    dfloat *d_uz = nullptr;
    dfloat *d_mxx = nullptr;
    dfloat *d_mxy = nullptr;
    dfloat *d_mxz = nullptr;
    dfloat *d_myy = nullptr;
    dfloat *d_myz = nullptr;
    dfloat *d_mzz = nullptr;
};

struct RemotePopulationHalos
{
    dfloat *back = nullptr;
    dfloat *front = nullptr;
    dfloat *initial_wall = nullptr;
};

struct PopulationHaloBuffers
{
    dfloat *send = nullptr;
    dfloat *periodic_send = nullptr;
    dfloat *color_send = nullptr;
    RemotePopulationHalos remote;
};

struct DeviceDomain
{
    int device_id;
    int subdomain_id;

    SubDomainInfo info;

    LBMState state;
    LBMState next_state;
    unsigned int active_moment_buffer;
    unsigned int *d_node_type;
    unsigned int *h_node_type;
    dfloat *d_output_staging;
    dfloat *h_output_staging;
    PopulationHaloBuffers population_halos;

    cudaStream_t stream;
    cudaStream_t interior_stream;
    cudaEvent_t interior_complete;
    cudaEvent_t boundary_complete;

    DeviceDomain(
        int physical_device_id,
        int logical_subdomain_id,
        const SubDomainInfo &subdomain_info)
        : device_id(physical_device_id),
          subdomain_id(logical_subdomain_id),
          info(subdomain_info),
          active_moment_buffer(0),
          d_node_type(nullptr),
          h_node_type(nullptr),
          d_output_staging(nullptr),
          h_output_staging(nullptr),
          stream(nullptr),
          interior_stream(nullptr),
          interior_complete(nullptr),
          boundary_complete(nullptr)
    {
    }
};

#endif
