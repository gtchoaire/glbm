
#ifndef STATISTICS_CONFIG_H
#define STATISTICS_CONFIG_H

#include <cstddef>

constexpr bool STATS_ENABLED = STATISTICS_ENABLED_VALUE != 0;

constexpr double STATS_START_T_STAR = STATISTICS_START_T_STAR_VALUE;

constexpr double STATS_END_T_STAR = STATISTICS_END_T_STAR_VALUE;

constexpr std::size_t STATS_SAMPLE_INTERVAL = STATISTICS_SAMPLE_INTERVAL_VALUE;

constexpr std::size_t BMBC_SAMPLE_INTERVAL = BMBC_SAMPLE_INTERVAL_VALUE;

static_assert(STATS_START_T_STAR >= 0.0, "Statistics start time must be non-negative");

static_assert(STATS_END_T_STAR > STATS_START_T_STAR, "Statistics end time must exceed start time");

static_assert(STATS_SAMPLE_INTERVAL > 0, "Statistics sample interval must be positive");

static_assert(BMBC_SAMPLE_INTERVAL > 0, "BMBC sample interval must be positive");

#endif
