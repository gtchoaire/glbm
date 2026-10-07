#ifndef MODEL
#define MODEL

#include <limits>
#include "var.h"

constexpr dfloat RE = RE_NUMBER;
constexpr int SCALE = 1;
constexpr int N = N_SIZE * SCALE;

constexpr int GLOBAL_NX = N;
constexpr int GLOBAL_NY = N;
constexpr int GLOBAL_NZ = SAR * N;

constexpr dfloat U_MAX = 0.05;
constexpr dfloat L = N;
constexpr dfloat RHO_0 = 1.0;
constexpr dfloat VISC = U_MAX * (GLOBAL_NX - 1) / RE;
constexpr dfloat TAU = 0.5 + 3.0 * VISC;

static_assert(
    GLOBAL_NZ % NUM_GPUS == 0,
    "GLOBAL_NZ must be divisible by NUM_GPUS");

constexpr int LOCAL_NX = GLOBAL_NX;
constexpr int LOCAL_NY = GLOBAL_NY;
constexpr int LOCAL_NZ = GLOBAL_NZ / NUM_GPUS;

constexpr int NX = LOCAL_NX;
constexpr int NY = LOCAL_NY;
constexpr int NZ = LOCAL_NZ;

constexpr int NZ_TOTAL = GLOBAL_NZ;

#ifndef SIMULATION_STEPS
constexpr int N_STEPS = 1;
#else
constexpr int N_STEPS = SIMULATION_STEPS;
#endif

#endif