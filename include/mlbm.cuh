#ifndef MLBM_H
#define MLBM_H

#include "globalFunctions.cuh"
#include "var.h"

#include PROPERTIES
#include COLLISION
#include RECONSTRUCTION

__global__ void gpuMomCollisionStreamAB(
    LBMState source,
    LBMState destination,
    const unsigned int *dNodeType,
    RemotePopulationHalos remote_halos,
    bool has_back_wall, bool has_front_wall,
    int global_z_start, unsigned int block_z_offset,
    unsigned int block_z_stride, bool initial_step);

#endif