#pragma once

#include "Model.h"

#include <cstddef>
#include <cstdint>

namespace neural_amp {

struct BenchmarkOptions {
    std::size_t samples = 10000;
    std::size_t warmupSamples = 1000;
    std::uint32_t seed = 1337;
};

struct BenchmarkResult {
    std::size_t samples = 0;
    double meanMicroseconds = 0.0;
    double p50Microseconds = 0.0;
    double p95Microseconds = 0.0;
    double p99Microseconds = 0.0;
    double maxMicroseconds = 0.0;
    double samplesPerSecond = 0.0;
    float checksum = 0.0f;
};

BenchmarkResult benchmarkStreamingInference(const NeuralNet& net, const BenchmarkOptions& options);

} // namespace neural_amp
