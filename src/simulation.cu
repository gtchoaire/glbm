#include "../include/simulation.cuh"

#include <cstdio>
#include <cuda_runtime.h>
#include <utility>
#include "../include/definitions.h"
#include "../include/lbmState.cuh"
#include "../include/mlbm.cuh"
#include "../include/nodeType.h"
#include "../include/output.h"

namespace {

double calculateMlups(const size_t nodes, const size_t steps, const double seconds) 
{
  return static_cast<double>(nodes) * static_cast<double>(steps) / (1.0e6 * seconds);
}

void reportProgress(const int completed_steps, const double solver_seconds) {
  const double progress = 100.0 * static_cast<double>(completed_steps) / static_cast<double>(N_STEPS);
  const double remaining_seconds = solver_seconds * static_cast<double>(N_STEPS - completed_steps) / static_cast<double>(completed_steps);
  const auto eta = static_cast<unsigned long long>(remaining_seconds + 0.5);
  const auto eta_hours = eta / 3600;
  const auto eta_minutes = (eta % 3600) / 60;
  const auto eta_seconds = eta % 60;

  std::printf("Step %010d / %010d | %6.2f%% | "
              "MLUPS %6.0f | "
              "ETA %02llu:%02llu:%02llu\n",
              completed_steps, N_STEPS, progress,
              calculateMlups(NUMBER_LBM_NODES, completed_steps, solver_seconds),
              eta_hours, eta_minutes, eta_seconds);
}
} // namespace

void runSimulation() 
{
  int available_gpus = 0;
  checkCuda(cudaGetDeviceCount(&available_gpus));

  if (GPU_INDEX >= available_gpus) {
    std::printf("GPU index %d is unavailable; detected %d device(s).\n",
                GPU_INDEX, available_gpus);
    std::exit(EXIT_FAILURE);
  }

  checkCuda(cudaSetDevice(GPU_INDEX));

  const SubDomainInfo subdomain(0, 0, GLOBAL_NZ - 1, LOCAL_NZ, GLOBAL_NZ, true,
                                true);
  DeviceDomain domain(0, 0, subdomain);

  allocateLBMState(domain.state);
  allocateLBMState(domain.next_state);
  if (!initializeNodeTypes(domain)) {
    std::printf("Failed to allocate the node type map.\n");
    std::exit(EXIT_FAILURE);
  }

  const dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
  const dim3 grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);
  initializeEquilibrium(domain.state, grid, block);
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&domain.population_halos.remote.initial_wall), Q * sizeof(dfloat)));
  gpuInitializeWallPopulations<<<1, 1>>>(domain.state, domain.population_halos.remote.initial_wall);
  checkCuda(cudaGetLastError());

  LBMState *current = &domain.state;
  LBMState *next = &domain.next_state;

  cudaEvent_t solver_start = nullptr;
  cudaEvent_t solver_stop = nullptr;
  checkCuda(cudaEventCreate(&solver_start));
  checkCuda(cudaEventCreate(&solver_stop));

  double solver_seconds = 0.0;
  double solver_seconds_at_last_report = 0.0;
  int completed_steps = 0;
  checkCuda(cudaEventRecord(solver_start));

  for (size_t step = 0; step < N_STEPS; ++step) 
  {
    if (step == 0) {
      gpuMomCollisionStreamAB<<<grid, block>>>(*current, *next, domain.d_node_type, domain.population_halos.remote, true);
      checkCuda(cudaGetLastError());
      gpuCollideMoments<<<grid, block>>>(*next);
      checkCuda(cudaGetLastError());

    } else {
      gpuStreamMomentsAB<<<grid, block>>>(*current, *next, domain.d_node_type, domain.population_halos.remote);
      checkCuda(cudaGetLastError());
    }

    std::swap(current, next);

    completed_steps = step + 1;
    const bool save_output = SAVE_VTI_OUTPUT && (completed_steps % OUTPUT_INTERVAL == 0 || completed_steps == N_STEPS);
    const bool print_progress = completed_steps % PRINT_INTERVAL == 0 || completed_steps == N_STEPS;
    const bool measurement_boundary = save_output || print_progress;

    if (measurement_boundary) 
    {
      float segment_milliseconds = 0.0F;
      checkCuda(cudaEventRecord(solver_stop));
      checkCuda(cudaEventSynchronize(solver_stop));
      checkCuda(cudaEventElapsedTime(&segment_milliseconds, solver_start, solver_stop));
      solver_seconds += static_cast<double>(segment_milliseconds) * 1.0e-3;
    }

    if (save_output) 
    {
      copyMacroscopicFieldsToHost(*current);
      writeVti(*current, completed_steps);
    }

    if (print_progress) 
    {
      const double interval_solver_seconds = solver_seconds - solver_seconds_at_last_report;

      reportProgress(completed_steps, solver_seconds);

      solver_seconds_at_last_report = solver_seconds;
    }
    if (measurement_boundary && completed_steps < N_STEPS)
      checkCuda(cudaEventRecord(solver_start));
  }

  if (solver_stop != nullptr)
    cudaEventDestroy(solver_stop);
  if (solver_start != nullptr)
    cudaEventDestroy(solver_start);
  if (domain.population_halos.remote.initial_wall != nullptr)
    cudaFree(domain.population_halos.remote.initial_wall);

  freeNodeTypes(domain);
  freeLBMState(domain.state);
  freeLBMState(domain.next_state);
}
