#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gpu_lab {

struct Summary {
    double median_ms{};
    double mean_ms{};
    double min_ms{};
    double std_ms{};
};

inline Summary summarize(std::vector<double> samples_ms) {
    if (samples_ms.empty()) {
        throw std::invalid_argument("at least one timing sample is required");
    }
    const double mean = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
                        static_cast<double>(samples_ms.size());
    double variance = 0.0;
    for (const double value : samples_ms) {
        const double delta = value - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(samples_ms.size());
    std::sort(samples_ms.begin(), samples_ms.end());
    const std::size_t middle = samples_ms.size() / 2;
    const double median = (samples_ms.size() % 2 == 0)
                              ? (samples_ms[middle - 1] + samples_ms[middle]) * 0.5
                              : samples_ms[middle];
    return {median, mean, samples_ms.front(), std::sqrt(variance)};
}

inline double effective_bandwidth_gbps(std::size_t bytes_moved, double milliseconds) {
    return milliseconds > 0.0
               ? static_cast<double>(bytes_moved) / (milliseconds * 1.0e6)
               : 0.0;
}

}  // namespace gpu_lab

