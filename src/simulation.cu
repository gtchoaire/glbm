#include "../include/simulation.cuh"
#include <chrono>
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

namespace
{
template<bool Enabled, size_t Interval>
constexpr bool outputDue(const size_t completed_steps)
{
  if constexpr (!Enabled)
    return false;
  else
  {
    static_assert(Interval > 0, "Enabled output interval must be positive");
    return completed_steps % Interval == 0 || completed_steps == static_cast<size_t>(N_STEPS);
  }
}
}

constexpr double calculateMlups(const size_t nodes, const size_t steps, const double seconds)
{
  return static_cast<double>(nodes) * static_cast<double>(steps) / (1.0e6 * seconds);
}

void reportProgress(const size_t completed_steps, const size_t executed_steps, const double wall_seconds, const double device_seconds)
{
  const double progress = 100.0 * static_cast<double>(completed_steps) / static_cast<double>(N_STEPS);
  const double remaining_seconds = wall_seconds * static_cast<double>(N_STEPS - completed_steps) / static_cast<double>(executed_steps);
  const auto eta = static_cast<unsigned long long>(remaining_seconds + 0.5);
  const auto eta_hours = eta / 3600;
  const auto eta_minutes = (eta % 3600) / 60;
  const auto eta_seconds = eta % 60;

  const double wall_mlups = calculateMlups(NUMBER_LBM_NODES * NUM_GPUS, executed_steps, wall_seconds);
  const double device_mlups = calculateMlups(NUMBER_LBM_NODES * NUM_GPUS, executed_steps, device_seconds);

  std::printf("\rStep %010zu / %010d | %6.2f%% | "
              "Wall MLUPS %6.0f | "
              "Device MLUPS %6.0f | "
              "ETA %02llu:%02llu:%02llu",
              completed_steps, N_STEPS, progress,
              wall_mlups, device_mlups,
              eta_hours, eta_minutes, eta_seconds);
  std::fflush(stdout);
}


void runSimulation() 
{
  if (!validateMultiGpuTopology())
    throw std::runtime_error("Invalid CUDA device configuration");

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
    domains.emplace_back(std::make_unique<DeviceDomain>(info.gpu_id, subdomain_id, info));
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
  checkCuda(cudaSetDevice(domains.at(0)->device_id));
  checkCuda(cudaEventCreate(&solver_start));
  checkCuda(cudaEventCreate(&solver_stop));

  size_t completed_steps = 0;
  std::filesystem::path output_directory;
  if constexpr (RESTART_ENABLED)
    completed_steps = loadCheckpoint(domains, current, RESTART_PATH, output_directory);
  if (completed_steps > static_cast<size_t>(N_STEPS))
    throw std::runtime_error("Checkpoint step is greater than the configured final step");

  if (output_directory.empty()) {
    if (!createOutputDirectory(output_directory) || !writeSimulationInfo(output_directory)) std::exit(EXIT_FAILURE);
  } else if (!std::filesystem::is_directory(output_directory)) {
    throw std::runtime_error("Checkpoint output directory does not exist: " + output_directory.string());
  }

  if constexpr (RESTART_ENABLED)
    reconcileKineticEnergyOutput(output_directory, completed_steps);

  const std::filesystem::path checkpoint_filename(CHECKPOINT_PATH);
  if constexpr (CHECKPOINT_ENABLED)
    if (checkpoint_filename.empty() || checkpoint_filename.has_parent_path())
      throw std::runtime_error("checkpoint_path must contain only a file name, for example 'checkpoint.glbm'");
  const std::filesystem::path checkpoint_output_path = output_directory / checkpoint_filename;

  const size_t initial_step = completed_steps;

  // Initializing clock variables

  double wall_seconds = 0.0;
  double device_seconds = 0.0;
  checkCuda(cudaSetDevice(domains.at(0)->device_id));
  auto wall_start = std::chrono::steady_clock::now();
  checkCuda(cudaEventRecord(solver_start));

  // Main loop

  for (size_t step = completed_steps; step < N_STEPS; ++step)
  {
    if constexpr (NUM_GPUS == 2) {
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
        if constexpr (NUM_GPUS == 2) {
            checkCuda(cudaEventRecord(domain.interior_complete, domain.stream));
            checkCuda(cudaEventRecord(domain.boundary_complete, domain.stream));
        }
      } else if constexpr (NUM_GPUS == 2) {
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
    const bool save_vti = outputDue<SAVE_VTI_OUTPUT, OUTPUT_INTERVAL>(completed_steps);
    const bool save_binary = outputDue<SAVE_BINARY_OUTPUT, BINARY_OUTPUT_INTERVAL_VALUE>(completed_steps);
    const bool save_kinetic_energy = outputDue<SAVE_KINETIC_ENERGY_OUTPUT, KINETIC_ENERGY_INTERVAL>(completed_steps);
    const bool save_checkpoint = outputDue<CHECKPOINT_ENABLED, CHECKPOINT_INTERVAL>(completed_steps);
    const bool print_progress = completed_steps % PRINT_INTERVAL == 0 || completed_steps == N_STEPS;
    const bool measurement = save_vti || save_binary || save_kinetic_energy || save_checkpoint || print_progress;


    if (measurement) 
    {
      for (const auto &domain : domains) {
        checkCuda(cudaSetDevice(domain->device_id));
        checkCuda(cudaDeviceSynchronize());
      }

      checkCuda(cudaSetDevice(domains.at(0)->device_id));
      checkCuda(cudaEventRecord(solver_stop));
      checkCuda(cudaEventSynchronize(solver_stop));

      float segment_milliseconds = 0.0f;

      checkCuda(cudaEventElapsedTime(&segment_milliseconds, solver_start, solver_stop));

      // Device timing
      device_seconds += static_cast<double>(segment_milliseconds) * 1.0e-3;

      // Wall-clock timing
      auto wall_stop = std::chrono::steady_clock::now();
      wall_seconds += std::chrono::duration<double>(wall_stop - wall_start).count();

      // std::printf("Milli and seconds | %6.6f | %6.6f |  \n",
      //             segment_milliseconds, wall_seconds);
    }

    if (save_vti || save_binary || save_kinetic_energy || save_checkpoint)
    {
      for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id) {
        checkCuda(cudaSetDevice(domains[subdomain_id]->device_id));
        copyMacroscopicFieldsToHost(*current[subdomain_id]);
      }
    }

    if constexpr (SAVE_VTI_OUTPUT)
      if (save_vti && !writeVti(current, completed_steps, output_directory))
        std::exit(EXIT_FAILURE);

    if constexpr (SAVE_BINARY_OUTPUT)
      if (save_binary && !writeBinaryFields(current, completed_steps, output_directory))
        std::exit(EXIT_FAILURE);

    if constexpr (CHECKPOINT_ENABLED)
      if (save_checkpoint)
        writeCheckpoint(current, completed_steps, checkpoint_output_path.string(), output_directory);

    if constexpr (SAVE_KINETIC_ENERGY_OUTPUT)
      if (save_kinetic_energy)
        totalKineticEnergy(current, completed_steps, output_directory);

    if (print_progress) 
      reportProgress(completed_steps, completed_steps - initial_step, wall_seconds, device_seconds);

    if (measurement && completed_steps < N_STEPS)
    {
      checkCuda(cudaSetDevice(domains.at(0)->device_id));
      wall_start = std::chrono::steady_clock::now();
      checkCuda(cudaEventRecord(solver_start));
    }
  }

  std::printf("\n");

  checkCuda(cudaSetDevice(domains.at(0)->device_id));
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
