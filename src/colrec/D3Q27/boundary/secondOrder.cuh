#ifndef SECOND_ORDER_BOUNDARY_CUH
#define SECOND_ORDER_BOUNDARY_CUH

#include "../../../../include/nodeType.h"
#include "../../../../include/var.h"

__device__ inline void apply_boundaries(
    dfloat *pop, dfloat *rhoVar,
    dfloat *ux, dfloat *uy, dfloat *uz,
    dfloat *mxx, dfloat *mxy, dfloat *mxz,
    dfloat *myy, dfloat *myz, dfloat *mzz,
    unsigned int nodeType)
{
    switch (nodeType)
    {
    case SOUTH_WEST_BACK:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[20];

        const dfloat rho = static_cast<dfloat>(12) * rho_I / static_cast<dfloat>(7);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_WEST_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18] + pop[22];

        const dfloat rho = static_cast<dfloat>(12) * rho_I / static_cast<dfloat>(7);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_EAST_BACK:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];

        const dfloat rho = static_cast<dfloat>(12) * rho_I / static_cast<dfloat>(7);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_EAST_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];

        const dfloat rho = static_cast<dfloat>(12) * rho_I / static_cast<dfloat>(7);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_WEST:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = (pop[8] + pop[20] + pop[22]) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = (static_cast<dfloat>(36) * mxy_I * rho_I - rho) / (static_cast<dfloat>(9) * rho);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_EAST:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = -(pop[13] + pop[23] + pop[26]) * inv_rho_I;

        const dfloat rho = -static_cast<dfloat>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) / (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = (static_cast<dfloat>(36) * mxy_I * rho_I + rho) / (static_cast<dfloat>(9) * rho);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case WEST_BACK:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxz_I = (pop[10] + pop[20] + pop[24]) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = (static_cast<dfloat>(36) * mxz_I * rho_I - rho) / (static_cast<dfloat>(9) * rho);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case WEST_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxz_I = -(pop[16] + pop[22] + pop[25]) * inv_rho_I;

        const dfloat rho = -static_cast<dfloat>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = (static_cast<dfloat>(36) * mxz_I * rho_I + rho) / (static_cast<dfloat>(9) * rho);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case EAST_BACK:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxz_I = -(pop[15] + pop[21] + pop[26]) * inv_rho_I;

        const dfloat rho = -static_cast<dfloat>(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = (static_cast<dfloat>(36) * mxz_I * rho_I + rho) / (static_cast<dfloat>(9) * rho);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case EAST_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxz_I = (pop[9] + pop[19] + pop[23]) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = (static_cast<dfloat>(36) * mxz_I * rho_I - rho) / (static_cast<dfloat>(9) * rho);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_BACK:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[20] + pop[26];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat myz_I = (pop[12] + pop[20] + pop[26]) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = (static_cast<dfloat>(36) * myz_I * rho_I - rho) / (static_cast<dfloat>(9) * rho);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18] + pop[22] + pop[23];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat myz_I = -(pop[18] + pop[22] + pop[23]) * inv_rho_I;

        const dfloat rho = -static_cast<dfloat>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = (static_cast<dfloat>(36) * myz_I * rho_I + rho) / (static_cast<dfloat>(9) * rho);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case WEST:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18] + pop[20] + pop[22] + pop[24] + pop[25];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const dfloat mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(6) * rho_I / static_cast<dfloat>(5);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(2) * mxy_I * rho_I / rho;
        *mxz = static_cast<dfloat>(2) * mxz_I * rho_I / rho;
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case EAST:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const dfloat mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(6) * rho_I / static_cast<dfloat>(5);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(2) * mxy_I * rho_I / rho;
        *mxz = static_cast<dfloat>(2) * mxz_I * rho_I / rho;
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case SOUTH:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const dfloat myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(6) * rho_I / static_cast<dfloat>(5);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(2) * mxy_I * rho_I / rho;
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(2) * myz_I * rho_I / rho;
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case BACK:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const dfloat myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(6) * rho_I / static_cast<dfloat>(5);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(2) * mxz_I * rho_I / rho;
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(2) * myz_I * rho_I / rho;
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case FRONT:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;
        const dfloat myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(6) * rho_I / static_cast<dfloat>(5);

        *rhoVar = rho;
        *ux = static_cast<dfloat>(0);
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = static_cast<dfloat>(0);
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(2) * mxz_I * rho_I / rho;
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(2) * myz_I * rho_I / rho;
        *mzz = static_cast<dfloat>(0);

        return;
    }

    case NORTH:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const dfloat myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(6) * rho_I / static_cast<dfloat>(5);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = (static_cast<dfloat>(6) * mxy_I * rho_I - U_MAX * rho) / (static_cast<dfloat>(3) * rho);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(2) * myz_I * rho_I / rho;
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_WEST_BACK:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17] + pop[24];

        const dfloat rho = -static_cast<dfloat>(216) * rho_I /
                           (-static_cast<dfloat>(125) - static_cast<dfloat>(75) * U_MAX + static_cast<dfloat>(75) * U_MAX * U_MAX);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_WEST_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16] + pop[25];

        const dfloat rho = -static_cast<dfloat>(216) * rho_I /
                           (-static_cast<dfloat>(125) - static_cast<dfloat>(75) * U_MAX + static_cast<dfloat>(75) * U_MAX * U_MAX);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_EAST_BACK:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];

        const dfloat rho = -static_cast<dfloat>(216) * rho_I /
                           (-static_cast<dfloat>(125) + static_cast<dfloat>(75) * U_MAX + static_cast<dfloat>(75) * U_MAX * U_MAX);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_EAST_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];

        const dfloat rho = -static_cast<dfloat>(216) * rho_I /
                           (-static_cast<dfloat>(125) + static_cast<dfloat>(75) * U_MAX + static_cast<dfloat>(75) * U_MAX * U_MAX);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_BACK:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat myz_I = -(pop[17] + pop[21] + pop[24]) * inv_rho_I;

        const dfloat rho = -static_cast<dfloat>(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = (static_cast<dfloat>(36) * myz_I * rho_I + rho) / (static_cast<dfloat>(9) * rho);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_FRONT:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16] + pop[19] + pop[25];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat myz_I = (pop[11] + pop[19] + pop[25]) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + OMEGA);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = static_cast<dfloat>(0);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = (static_cast<dfloat>(36) * myz_I * rho_I - rho) / (static_cast<dfloat>(9) * rho);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_EAST:
    {
        const dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = (pop[7] + pop[19] + pop[21]) * inv_rho_I;

        const dfloat rho = static_cast<dfloat>(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) - static_cast<dfloat>(18) * U_MAX - static_cast<dfloat>(18) * U_MAX * U_MAX + OMEGA + static_cast<dfloat>(3) * U_MAX * OMEGA + static_cast<dfloat>(3) * U_MAX * U_MAX * OMEGA);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = (static_cast<dfloat>(36) * mxy_I * rho_I - rho - static_cast<dfloat>(3) * U_MAX * rho - static_cast<dfloat>(3) * U_MAX * U_MAX * rho) /
               (static_cast<dfloat>(9) * rho);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    case NORTH_WEST:
    {
        const dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[24] + pop[25];
        const dfloat inv_rho_I = static_cast<dfloat>(1) / rho_I;

        const dfloat mxy_I = -(pop[14] + pop[24] + pop[25]) * inv_rho_I;

        const dfloat rho = -static_cast<dfloat>(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) /
                           (static_cast<dfloat>(24) + static_cast<dfloat>(18) * U_MAX - static_cast<dfloat>(18) * U_MAX * U_MAX + OMEGA - static_cast<dfloat>(3) * U_MAX * OMEGA + static_cast<dfloat>(3) * U_MAX * U_MAX * OMEGA);

        *rhoVar = rho;
        *ux = U_MAX;
        *uy = static_cast<dfloat>(0);
        *uz = static_cast<dfloat>(0);
        *mxx = U_MAX * U_MAX;
        *mxy = (static_cast<dfloat>(36) * mxy_I * rho_I + rho - static_cast<dfloat>(3) * U_MAX * rho + static_cast<dfloat>(3) * U_MAX * U_MAX * rho) /
               (static_cast<dfloat>(9) * rho);
        *mxz = static_cast<dfloat>(0);
        *myy = static_cast<dfloat>(0);
        *myz = static_cast<dfloat>(0);
        *mzz = static_cast<dfloat>(0);

        return;
    }
    }
}

#endif
