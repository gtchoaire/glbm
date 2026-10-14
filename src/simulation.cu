#include "../include/simulation.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../include/auxFunctions.cuh"
#include "../include/checkpoint.h"
#include "../include/definitions.h"
#include "../include/lbmState.cuh"
#include "../include/mlbm.cuh"
#include "../include/multiGpu.cuh"
#include "../include/nodeType.h"
#include "../include/output.h"



double calculateMlups(const size_t nodes, const size_t steps, const double seconds) 
{
  return static_cast<double>(nodes) * static_cast<double>(steps) / (1.0e6 * seconds);
}

void reportProgress(
  const size_t completed_steps,
  const size_t executed_steps,
  const double solver_seconds) {
  const double progress = 100.0 * static_cast<double>(completed_steps) / static_cast<double>(N_STEPS);
  const double remaining_seconds = solver_seconds * static_cast<double>(N_STEPS - completed_steps) / static_cast<double>(executed_steps);
  const auto eta = static_cast<unsigned long long>(remaining_seconds + 0.5);
  const auto eta_hours = eta / 3600;
  const auto eta_minutes = (eta % 3600) / 60;
  const auto eta_seconds = eta % 60;

  std::printf("\rStep %010zu / %010d | %6.2f%% | "
              "MLUPS %6.0f | "
              "ETA %02llu:%02llu:%02llu",
              completed_steps, N_STEPS, progress,
              calculateMlups(NUMBER_LBM_NODES * NUM_GPUS, executed_steps, solver_seconds),
              eta_hours, eta_minutes, eta_seconds);
  std::fflush(stdout);
}


void runSimulation() 
{

  std::filesystem::path output_directory;
  if (!createOutputDirectory(output_directory) ||
      !writeSimulationInfo(output_directory))
    std::exit(EXIT_FAILURE);


  const dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
  const dim3 grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);
  std::vector<std::unique_ptr<DeviceDomain>> domains;
  std::vector<LBMState *> current;
  std::vector<LBMState *> next;
  domains.reserve(NUM_GPUS);
  current.reserve(NUM_GPUS);
  next.reserve(NUM_GPUS);

  for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id) {
    const SubDomainInfo info = makeSubDomainInfo(subdomain_id);
    domains.emplace_back(std::make_unique<DeviceDomain>(
      info.gpu_id, subdomain_id, info));
    DeviceDomain &domain = *domains.back();
    checkCuda(cudaSetDevice(domain.device_id));
    allocateLBMState(domain.state);
    allocateLBMState(domain.next_state);
    if (!initializeNodeTypes(domain))
      throw std::runtime_error("Failed to initialize node types");

    allocateMomentHalos(domain);
    initializeEquilibrium(domain.state, grid, block);
    gpuInitializeWallPopulations<<<1, 1>>>(domain.state, domain.population_halos.remote.initial_wall);
    checkCuda(cudaGetLastError());
    current.push_back(&domain.state);
    next.push_back(&domain.next_state);
  }

  cudaEvent_t solver_start = nullptr;
  cudaEvent_t solver_stop = nullptr;
  checkCuda(cudaSetDevice(domains.front()->device_id));
  checkCuda(cudaEventCreate(&solver_start));
  checkCuda(cudaEventCreate(&solver_stop));

  size_t completed_steps = 0;
  if (RESTART_PATH[0] != '\0')
    completed_steps = loadCheckpoint(domains, current, RESTART_PATH);
  if (completed_steps > static_cast<size_t>(N_STEPS))
    throw std::runtime_error(
      "Checkpoint step is greater than the configured final step");
  const size_t initial_step = completed_steps;

  double solver_seconds = 0.0;
  double solver_seconds_at_last_report = 0.0;
  checkCuda(cudaSetDevice(domains.front()->device_id));
  checkCuda(cudaEventRecord(solver_start));

  for (size_t step = completed_steps; step < N_STEPS; ++step)
  {
    if (NUM_GPUS == 2) {
      beginMomentHaloExchange(*domains[0], *domains[1], *current[0], *current[1]);

      if (step > 0) {
        const dim3 interior_grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z - 2);
        for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id) {
          DeviceDomain &domain = *domains[subdomain_id];
          checkCuda(cudaSetDevice(domain.device_id));
          gpuStreamMomentsAB<<<interior_grid, block, 0, domain.interior_stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 1);
          checkCuda(cudaGetLastError());
        }
      }

      completeMomentHaloExchange(*domains[0], *domains[1]);
    }

    for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id) {
      DeviceDomain &domain = *domains[subdomain_id];
      checkCuda(cudaSetDevice(domain.device_id));
      if (step == 0) {
        const cudaStream_t launch_stream = NUM_GPUS == 2 ? domain.stream : nullptr;
        gpuBootstrapABState<<<grid, block, 0, launch_stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 0);
        checkCuda(cudaGetLastError());
        gpuCollideMoments<<<grid, block, 0, launch_stream>>>(*next[subdomain_id]);
        checkCuda(cudaGetLastError());
        if (NUM_GPUS == 2) {
            checkCuda(cudaEventRecord(domain.interior_complete, domain.stream));
            checkCuda(cudaEventRecord(domain.boundary_complete, domain.stream));
        }
      } else if (NUM_GPUS == 2) {
        const dim3 boundary_grid(NUM_BLOCK_X, NUM_BLOCK_Y, 1);
        gpuStreamMomentsAB<<<boundary_grid, block, 0, domain.stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 0);
        checkCuda(cudaGetLastError());
        gpuStreamMomentsAB<<<boundary_grid, block, 0, domain.stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, NUM_BLOCK_Z - 1);
        checkCuda(cudaGetLastError());
        checkCuda(cudaEventRecord(domain.interior_complete, domain.interior_stream));
        checkCuda(cudaEventRecord(domain.boundary_complete, domain.stream));
      } else {
        gpuStreamMomentsAB<<<grid, block>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 0);
        checkCuda(cudaGetLastError());
      }
      std::swap(current[subdomain_id], next[subdomain_id]);
    }

    completed_steps = step + 1;
    const bool save_vti = SAVE_VTI_OUTPUT && (completed_steps % OUTPUT_INTERVAL == 0 || completed_steps == N_STEPS);
    const bool save_binary = SAVE_BINARY_OUTPUT && (completed_steps % BINARY_OUTPUT_INTERVAL_VALUE == 0 || completed_steps == N_STEPS);
    const bool save_kinetic_energy = SAVE_KINETIC_ENERGY_OUTPUT &&
      (completed_steps % KINETIC_ENERGY_INTERVAL == 0 || completed_steps == N_STEPS);
    const bool save_checkpoint = CHECKPOINT_INTERVAL > 0 && (completed_steps % CHECKPOINT_INTERVAL == 0 || completed_steps == N_STEPS);
    const bool print_progress = completed_steps % PRINT_INTERVAL == 0 || completed_steps == N_STEPS;
    const bool measurement = save_vti || save_binary || save_kinetic_energy ||
      save_checkpoint || print_progress;

    if (measurement) 
    {
      for (const auto &domain : domains) {
        checkCuda(cudaSetDevice(domain->device_id));
        checkCuda(cudaDeviceSynchronize());
      }
      float segment_milliseconds = 0.0F;
      checkCuda(cudaSetDevice(domains.front()->device_id));
      checkCuda(cudaEventRecord(solver_stop));
      checkCuda(cudaEventSynchronize(solver_stop));
      checkCuda(cudaEventElapsedTime(&segment_milliseconds, solver_start, solver_stop));
      solver_seconds += static_cast<double>(segment_milliseconds) * 1.0e-3;
    }

    if (save_vti || save_binary || save_kinetic_energy || save_checkpoint)
    {
      for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id) {
        checkCuda(cudaSetDevice(domains[subdomain_id]->device_id));
        copyMacroscopicFieldsToHost(*current[subdomain_id]);
      }
    }

    if (save_vti)
    {
      if (!writeVti(current, completed_steps, output_directory))
        std::exit(EXIT_FAILURE);
    }

    if (save_binary)
    {
      if (!writeBinaryFields(current, completed_steps, output_directory))
        std::exit(EXIT_FAILURE);
    }

    if (save_checkpoint)
      writeCheckpoint(current, completed_steps, CHECKPOINT_PATH);

    if (save_kinetic_energy)
      totalKineticEnergy(current, completed_steps, output_directory);

    if (print_progress) 
    {
      const double interval_solver_seconds = solver_seconds - solver_seconds_at_last_report;
      reportProgress(completed_steps, completed_steps - initial_step, solver_seconds);
      solver_seconds_at_last_report = solver_seconds;
    }
    if (measurement && completed_steps < N_STEPS)
    {
      checkCuda(cudaSetDevice(domains.front()->device_id));
      checkCuda(cudaEventRecord(solver_start));
    }
  }

  checkCuda(cudaSetDevice(domains.front()->device_id));
  cudaEventDestroy(solver_stop);
  cudaEventDestroy(solver_start);
  for (const auto &domain_ptr : domains) {
    DeviceDomain &domain = *domain_ptr;
    checkCuda(cudaSetDevice(domain.device_id));
    freeMomentHalos(domain);
    freeNodeTypes(domain);
    freeLBMState(domain.state);
    freeLBMState(domain.next_state);
  }
}
