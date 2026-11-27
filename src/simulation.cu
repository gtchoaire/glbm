#include "../include/simulation.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../include/globalStatistics.cuh"
#include "../include/checkpoint.h"
#include "../include/definitions.h"
#include "../include/lbmState.cuh"
#include "../include/mlbm.cuh"
#include "../include/multiGpu.cuh"
#include "../include/nodeType.h"
#include "../include/output.h"
#include "../include/localFieldStatistics.cuh"
#include "../include/localFieldStatisticsIO.h"
#include <algorithm>
#include "../include/statisticsConfig.h"
#include "../include/bmbc.cuh"
#include "../include/bmbcIO.cuh"

namespace
{
    template <bool Enabled, size_t Interval>
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

    template <size_t Interval>
    constexpr bool blockIntervalDue(const size_t completed_blocks)
    {
        if constexpr (Interval == 0)
            return false;
        else
            return completed_blocks >= Interval &&
                   completed_blocks % Interval == 0;
    }
}

constexpr double calculateMlups(const size_t nodes, const size_t steps, const double seconds)
{
    return static_cast<double>(nodes) * static_cast<double>(steps) / (1.0e6 * seconds);
}

void reportProgress(
    const size_t completed_steps,
    const size_t executed_steps,
    const double solver_wall_seconds,
    const double device_seconds,
    const double elapsed_seconds)
{
    const double progress = 100.0 * static_cast<double>(completed_steps) / static_cast<double>(N_STEPS);
    const double remaining_seconds = elapsed_seconds * static_cast<double>(N_STEPS - completed_steps) / static_cast<double>(executed_steps);
    const auto eta = static_cast<unsigned long long>(remaining_seconds + 0.5);
    const auto eta_hours = eta / 3600;
    const auto eta_minutes = (eta % 3600) / 60;
    const auto eta_seconds = eta % 60;

    const double wall_mlups = calculateMlups(NUMBER_PHYSICAL_LBM_NODES * NUM_GPUS, executed_steps, solver_wall_seconds);
    const double device_mlups = calculateMlups(NUMBER_PHYSICAL_LBM_NODES * NUM_GPUS, executed_steps, device_seconds);

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

    for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id)
    {
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
    std::filesystem::path run_directory;

    LocalFieldStatisticsSnapshot restored_local_fields;

    if constexpr (RESTART_ENABLED)
    {
        completed_steps = loadCheckpoint(
            domains,
            current,
            RESTART_PATH);
    }
    if (completed_steps > static_cast<size_t>(N_STEPS))
        throw std::runtime_error("Checkpoint step is greater than the configured final step");

    if (!prepareRunDirectory(run_directory))
        std::exit(EXIT_FAILURE);

    const std::filesystem::path checkpoint_directory = run_directory / "checkpoint";
    const std::filesystem::path statistics_directory = run_directory / "statistics";
    std::filesystem::create_directories(checkpoint_directory);
    std::filesystem::create_directories(statistics_directory);

    const std::filesystem::path checkpoint_filename(CHECKPOINT_PATH);
    if constexpr (CHECKPOINT_ENABLED)
        if (checkpoint_filename.empty() || checkpoint_filename.has_parent_path())
            throw std::runtime_error("checkpoint_path must contain only a file name, for example 'checkpoint.glbm'");
    const std::filesystem::path checkpoint_output_path = checkpoint_directory / checkpoint_filename;

    if constexpr (!RESTART_ENABLED)
        if (std::filesystem::exists(checkpoint_output_path) ||
            std::filesystem::exists(localFieldStatisticsPath(statistics_directory)) ||
            std::filesystem::exists(statistics_directory / "bmbc_blocks.bin"))
            throw std::runtime_error("Run directory already contains results; set restart_path or use another directory");

    if (!writeSimulationMetadata(statistics_directory))
        throw std::runtime_error("Failed to write simulation metadata");

    reconcileGlobalStatistics(statistics_directory, completed_steps);

    const size_t initial_step = completed_steps;

    // Convert dimensionless time into LBM steps.
    const auto timeToStep = [](const double t_star) -> size_t
    {
        const double steps =
            t_star * static_cast<double>(GLOBAL_NX - 1) /
            static_cast<double>(U_MAX);

        return static_cast<size_t>(std::ceil(steps));
    };

    const size_t stats_start_step =
        timeToStep(STATS_START_T_STAR);

    const size_t stats_end_step =
        timeToStep(STATS_END_T_STAR);

    const auto samplesThroughStep =
        [stats_start_step, stats_end_step](
            const size_t step, const size_t interval) -> size_t
    {
        if (step <= stats_start_step)
            return 0;
        const size_t last_step = std::min(step, stats_end_step);
        return (last_step - stats_start_step) / interval;
    };

    constexpr bool BMBC_ENABLED = BMBC_ENABLED_VALUE != 0;
    const size_t expected_bmbc_samples =
        samplesThroughStep(completed_steps, BMBC_SAMPLE_INTERVAL);

    if constexpr (STATS_ENABLED)
    {
        const size_t expected_samples =
            samplesThroughStep(completed_steps, STATS_SAMPLE_INTERVAL);

        if constexpr (RESTART_ENABLED)
            if (expected_samples > 0)
                restored_local_fields = readLocalFieldStatistics(
                    statistics_directory, completed_steps);

        if (restored_local_fields.samples > 0)
        {
            const bool extending_window =
                stats_end_step > restored_local_fields.end_step;
            const bool extension_has_gap =
                extending_window && completed_steps > restored_local_fields.end_step;
            if (restored_local_fields.start_step != stats_start_step ||
                restored_local_fields.sample_interval != STATS_SAMPLE_INTERVAL ||
                stats_end_step < restored_local_fields.end_step ||
                extension_has_gap)
            {
                throw std::runtime_error(
                    "Checkpoint statistics window is incompatible with case.toml. "
                    "The end may only be extended when restarting at or before "
                    "the previous statistics end.");
            }

            if (restored_local_fields.samples != expected_samples)
            {
                throw std::runtime_error(
                    "Checkpoint statistics sample count is inconsistent.");
            }
        }
        else if (expected_samples > 0)
        {
            throw std::runtime_error(
                "Checkpoint is missing accumulated statistics.");
        }
    }

    // Local field statistics allocation

    std::vector<LocalFieldStatistics> local_fields(NUM_GPUS);

    if constexpr (STATS_ENABLED)
    {
        for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
        {
            checkCuda(cudaSetDevice(domains[gpu]->device_id));

            allocateLocalFieldStatistics(local_fields[gpu]);
        }
    }

    if constexpr (STATS_ENABLED)
    {
        if (restored_local_fields.samples > 0)
        {
            if (restored_local_fields.means.size() !=
                static_cast<size_t>(NUM_GPUS))
            {
                throw std::runtime_error(
                    "Invalid number of statistics subdomains.");
            }

            for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
            {
                checkCuda(cudaSetDevice(domains[gpu]->device_id));

                restoreLocalFieldStatisticsFromHost(
                    local_fields[gpu],
                    restored_local_fields.means[gpu],
                    restored_local_fields.samples);
            }

            std::printf(
                "Local field statistics restored | samples %zu\n",
                restored_local_fields.samples);
        }
    }

    // BMBC statistical uncertainty monitoring. Profile sampling is independent
    // from the much more expensive local field statistics.

    BMBCProfiles bmbc;
    BMBCLocalProfileSum bmbc_local[2];
    BMBCMultiGPUExchange bmbc_exchange;

    if constexpr (BMBC_ENABLED)
    {
        checkCuda(cudaSetDevice(domains[0]->device_id));

        allocateBMBCProfiles(bmbc, static_cast<std::size_t>(BMBC_BLOCK_SAMPLES_VALUE), static_cast<std::size_t>(BMBC_INITIAL_CAPACITY_BLOCKS_VALUE));
        if constexpr (NUM_GPUS == 2)
        {
            allocateBMBCMultiGPUExchange(bmbc_exchange, domains[0]->device_id, domains[1]->device_id);
            for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
            {
                checkCuda(cudaSetDevice(domains[gpu]->device_id));
                allocateBMBCLocalProfileSum(bmbc_local[gpu]);
            }
            checkCuda(cudaSetDevice(domains[0]->device_id));
        }
        if constexpr (RESTART_ENABLED)
        {
            restoreBMBCSnapshot(
                bmbc,
                completed_steps,
                expected_bmbc_samples,
                stats_start_step,
                BMBC_SAMPLE_INTERVAL,
                statistics_directory);
        }
    }

    const auto saveResults = [&](const bool save_state)
    {
        std::vector<std::filesystem::path> results;

        if (save_state)
            results.push_back(writeCheckpoint(current, completed_steps, checkpoint_output_path.string()));

        if constexpr (STATS_ENABLED)
        {
            if (local_fields[0].samples > 0)
            {
                LocalFieldStatisticsSnapshot snapshot;
                snapshot.step = completed_steps;
                snapshot.start_step = stats_start_step;
                snapshot.end_step = stats_end_step;
                snapshot.sample_interval = STATS_SAMPLE_INTERVAL;
                snapshot.samples = local_fields[0].samples;
                snapshot.means.resize(NUM_GPUS);
                for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
                {
                    if (local_fields[gpu].samples != snapshot.samples)
                        throw std::runtime_error("Inconsistent samples between GPUs.");
                    checkCuda(cudaSetDevice(domains[gpu]->device_id));
                    copyLocalFieldStatisticsToHost(local_fields[gpu], snapshot.means[gpu]);
                }
                results.push_back(writeLocalFieldStatistics(snapshot, statistics_directory));
            }
        }

        if constexpr (BMBC_ENABLED)
        {
            if constexpr (NUM_GPUS == 2)
                flushBMBCMultiGPUProfiles(bmbc, bmbc_local[0], bmbc_local[1], bmbc_exchange, domains[0]->device_id, domains[1]->device_id, domains[0]->stream, domains[1]->stream);
            checkCuda(cudaSetDevice(domains[0]->device_id));
            results.push_back(writeBMBCSnapshot(
                bmbc, completed_steps, stats_start_step, BMBC_SAMPLE_INTERVAL,
                statistics_directory));
        }

        for (const std::filesystem::path &result : results)
            std::filesystem::rename(result.string() + ".tmp", result);
    };

    // Initializing clock variables

    double wall_seconds = 0.0;
    double device_seconds = 0.0;
    checkCuda(cudaSetDevice(domains.at(0)->device_id));
    const auto execution_start = std::chrono::steady_clock::now();
    auto wall_start = execution_start;
    checkCuda(cudaEventRecord(solver_start));

    // Main loop

    for (size_t step = completed_steps; step < N_STEPS; ++step)
    {
        if constexpr (NUM_GPUS == 2)
        {
            beginMomentHaloExchange(*domains[0], *domains[1], *current[0], *current[1]);

            if (step > 0)
            {
                const dim3 interior_grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z - 2);
                for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id)
                {
                    DeviceDomain &domain = *domains[subdomain_id];
                    checkCuda(cudaSetDevice(domain.device_id));
                    gpuStreamMomentsAB<<<interior_grid, block, 0, domain.interior_stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 1);
                    checkCuda(cudaGetLastError());
                }
            }

            completeMomentHaloExchange(*domains[0], *domains[1]);
        }

        for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id)
        {
            DeviceDomain &domain = *domains[subdomain_id];
            checkCuda(cudaSetDevice(domain.device_id));
            if (step == 0)
            {
                const cudaStream_t launch_stream = NUM_GPUS == 2 ? domain.stream : nullptr;
                gpuBootstrapABState<<<grid, block, 0, launch_stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 0);
                checkCuda(cudaGetLastError());
                gpuCollideMoments<<<grid, block, 0, launch_stream>>>(*next[subdomain_id]);
                checkCuda(cudaGetLastError());
                if constexpr (NUM_GPUS == 2)
                {
                    checkCuda(cudaEventRecord(domain.interior_complete, domain.stream));
                    checkCuda(cudaEventRecord(domain.boundary_complete, domain.stream));
                }
            }
            else if constexpr (NUM_GPUS == 2)
            {
                const dim3 boundary_grid(NUM_BLOCK_X, NUM_BLOCK_Y, 1);
                checkCuda(cudaStreamWaitEvent(domain.stream, domain.interior_complete, 0));
                gpuStreamMomentsAB<<<boundary_grid, block, 0, domain.stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 0);
                checkCuda(cudaGetLastError());
                gpuStreamMomentsAB<<<boundary_grid, block, 0, domain.stream>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, NUM_BLOCK_Z - 1);
                checkCuda(cudaGetLastError());
                checkCuda(cudaEventRecord(domain.interior_complete, domain.interior_stream));
                checkCuda(cudaEventRecord(domain.boundary_complete, domain.stream));
            }
            else
            {
                gpuStreamMomentsAB<<<grid, block>>>(*current[subdomain_id], *next[subdomain_id], domain.d_node_type, domain.population_halos.remote, domain.info, 0);
                checkCuda(cudaGetLastError());
            }
            std::swap(current[subdomain_id], next[subdomain_id]);
        }

        completed_steps = step + 1;

        const bool inside_stats_window =
            completed_steps > stats_start_step &&
            completed_steps <= stats_end_step;

        if constexpr (STATS_ENABLED)
        {
            const bool local_field_sample_due =
                inside_stats_window &&
                ((completed_steps - stats_start_step) %
                     STATS_SAMPLE_INTERVAL ==
                 0);

            if (local_field_sample_due)
            {
                for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
                {
                    DeviceDomain &domain = *domains[gpu];

                    checkCuda(cudaSetDevice(domain.device_id));

                    if constexpr (NUM_GPUS == 2)
                    {
                        checkCuda(cudaStreamWaitEvent(
                            domain.stream,
                            domain.interior_complete,
                            0));
                    }

                    accumulateLocalFieldStatistics(
                        local_fields[gpu],
                        *current[gpu],
                        grid,
                        block,
                        NUM_GPUS == 2 ? domain.stream : nullptr);
                }
            }
        }

        if constexpr (BMBC_ENABLED)
        {
            const bool profile_sample_due =
                inside_stats_window &&
                ((completed_steps - stats_start_step) %
                     BMBC_SAMPLE_INTERVAL ==
                 0);

            if (profile_sample_due)
            {
                if constexpr (NUM_GPUS == 2)
                {
                    for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
                    {
                        DeviceDomain &domain = *domains[gpu];
                        checkCuda(cudaSetDevice(domain.device_id));
                        checkCuda(cudaStreamWaitEvent(
                            domain.stream,
                            domain.interior_complete,
                            0));
                        accumulateBMBCLocalProfileSum(
                            bmbc_local[gpu], *current[gpu],
                            domain.info, domain.stream);
                    }
                    recordBMBCMultiGPUSample(
                        bmbc, bmbc_local[0], bmbc_local[1], bmbc_exchange,
                        domains[0]->device_id, domains[1]->device_id,
                        domains[0]->stream, domains[1]->stream);
                }
                else
                {
                    checkCuda(cudaSetDevice(domains[0]->device_id));
                    accumulateBMBCProfiles(bmbc, *current[0], domains[0]->info, nullptr);
                }

                constexpr std::size_t BMBC_REPORT_INTERVAL_BLOCKS =
                    BMBC_REPORT_INTERVAL_BLOCKS_VALUE;

                const bool completed_basic_block =
                    bmbc.samples_in_block == 0;
                const bool report_due =
                    blockIntervalDue<BMBC_REPORT_INTERVAL_BLOCKS>(
                        bmbc.completed_blocks);

                if (completed_basic_block && report_due)
                {
                    checkCuda(cudaSetDevice(domains[0]->device_id));

                    reportBMBCProfiles(
                        bmbc,
                        completed_steps,
                        NUM_GPUS == 2 ? domains[0]->stream : nullptr);
                }
            }
        }

        const bool save_vti = outputDue<SAVE_VTI_OUTPUT, OUTPUT_INTERVAL>(completed_steps);
        const bool save_binary = outputDue<SAVE_BINARY_OUTPUT, BINARY_OUTPUT_INTERVAL_VALUE>(completed_steps);
        const bool save_kinetic_energy = outputDue<SAVE_KINETIC_ENERGY_OUTPUT, KINETIC_ENERGY_INTERVAL>(completed_steps);
        const bool save_checkpoint = outputDue<CHECKPOINT_ENABLED, CHECKPOINT_INTERVAL>(completed_steps);
        const bool print_progress = completed_steps % PRINT_INTERVAL == 0 || completed_steps == N_STEPS;
        const bool measurement = save_vti || save_binary || save_kinetic_energy || save_checkpoint || print_progress;

        if (measurement)
        {
            for (const auto &domain : domains)
            {
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
        }

        if (save_vti || save_binary || save_checkpoint)
        {
            for (int subdomain_id = 0; subdomain_id < NUM_GPUS; ++subdomain_id)
            {
                checkCuda(cudaSetDevice(domains[subdomain_id]->device_id));
                copyMacroscopicFieldsToHost(*current[subdomain_id]);
            }
        }

        if constexpr (SAVE_VTI_OUTPUT)
            if (save_vti && !writeVti(current, completed_steps, statistics_directory / "instantaneous"))
                std::exit(EXIT_FAILURE);

        if constexpr (SAVE_BINARY_OUTPUT)
            if (save_binary && !writeBinaryFields(current, completed_steps, statistics_directory / "instantaneous"))
                std::exit(EXIT_FAILURE);

        if constexpr (CHECKPOINT_ENABLED)
        {
            if (save_checkpoint)
                saveResults(true);
        }

        if constexpr (SAVE_KINETIC_ENERGY_OUTPUT)
            if (save_kinetic_energy)
                writeGlobalStatistics(domains, current, completed_steps, statistics_directory);

        if (print_progress)
        {
            const double elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - execution_start).count();
            reportProgress(
                completed_steps,
                completed_steps - initial_step,
                wall_seconds,
                device_seconds,
                elapsed_seconds);
        }

        if (measurement && completed_steps < N_STEPS)
        {
            checkCuda(cudaSetDevice(domains.at(0)->device_id));
            wall_start = std::chrono::steady_clock::now();
            checkCuda(cudaEventRecord(solver_start));
        }
    }

    std::printf("\n");

    if constexpr (!CHECKPOINT_ENABLED)
        saveResults(false);

    if constexpr (BMBC_ENABLED)
    {
        if (bmbc.total_samples != samplesThroughStep(completed_steps, BMBC_SAMPLE_INTERVAL))
            throw std::runtime_error("BMBC sample count mismatch");

        std::printf(
            "BMBC | samples %zu | blocks %zu | partial %zu\n",
            bmbc.total_samples,
            bmbc.completed_blocks,
            bmbc.samples_in_block);

        freeBMBCProfiles(bmbc);
        if constexpr (NUM_GPUS == 2)
        {
            freeBMBCMultiGPUExchange(bmbc_exchange);
            for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
            {
                checkCuda(cudaSetDevice(domains[gpu]->device_id));
                freeBMBCLocalProfileSum(bmbc_local[gpu]);
            }
        }
    }

    // Release local field statistics buffers.
    if constexpr (STATS_ENABLED)
    {
        for (int gpu = 0; gpu < NUM_GPUS; ++gpu)
        {
            checkCuda(cudaSetDevice(domains[gpu]->device_id));
            freeLocalFieldStatistics(local_fields[gpu]);
        }
    }

    checkCuda(cudaSetDevice(domains.at(0)->device_id));
    cudaEventDestroy(solver_stop);
    cudaEventDestroy(solver_start);
    for (const auto &domain_ptr : domains)
    {
        DeviceDomain &domain = *domain_ptr;
        checkCuda(cudaSetDevice(domain.device_id));
        freeMomentHalos(domain);
        freeNodeTypes(domain);
        freeLBMState(domain.state);
        freeLBMState(domain.next_state);
    }
}
