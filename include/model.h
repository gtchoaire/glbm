#ifndef MODEL
#define MODEL

#include <limits>
#include "var.h"

constexpr dfloat RE = RE_NUMBER;
constexpr int SCALE = 1;
constexpr int N = N_SIZE * SCALE;
static_assert(N > 0 && SAR > 0, "Domain dimensions must be positive");

constexpr int GLOBAL_NX = N;
constexpr int GLOBAL_NY = N;
constexpr int GLOBAL_NZ = SAR * N;
static_assert(NUM_GPUS == 1 || NUM_GPUS == 2, "NUM_GPUS must be 1 or 2");
static_assert(GLOBAL_NZ % NUM_GPUS == 0, "GLOBAL_NZ must be divisible by NUM_GPUS");
static_assert(ORDER == 2, "Only second-order reconstruction is supported");

constexpr dfloat U_MAX = U_MAX_VALUE;
constexpr dfloat RHO_0 = 1.0;
static_assert(RE > 0, "RE must be positive");
static_assert(U_MAX > 0, "U_MAX must be positive");
constexpr dfloat VISC = U_MAX * (GLOBAL_NX - 1) / RE;
constexpr dfloat TAU = 0.5 + 3.0 * VISC;

constexpr int LOCAL_NX = GLOBAL_NX;
constexpr int LOCAL_NY = GLOBAL_NY;
constexpr int LOCAL_NZ = GLOBAL_NZ / NUM_GPUS;

constexpr int NX = LOCAL_NX;
constexpr int NY = LOCAL_NY;
constexpr int NZ = LOCAL_NZ;

#ifndef SIMULATION_STEPS
constexpr int N_STEPS = 1;
#else
constexpr int N_STEPS = SIMULATION_STEPS;
#endif

#endif
