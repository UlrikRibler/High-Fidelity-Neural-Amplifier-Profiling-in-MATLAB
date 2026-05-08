#include "Benchmark.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace stock_signal {
namespace {

double percentile(const std::vector<double>& sortedValues, double p) {
    if (sortedValues.empty()) {
        return 0.0;
    }
    const double clamped = std::clamp(p, 0.0, 1.0);
    const auto index = static_cast<std::size_t>(
        std::round(clamped * static_cast<double>(sortedValues.size() - 1)));
    return sortedValues[index];
}

Eigen::MatrixXf makeBenchmarkInput(const ModelConfig& cfg, std::size_t columns, std::uint32_t seed) {
    Eigen::MatrixXf input(cfg.inputSize, static_cast<int>(columns));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> noise(-0.01f, 0.01f);

    for (std::size_t t = 0; t < columns; ++t) {
        const float phase = static_cast<float>(t) * 0.0174532925f;
        input(0, static_cast<int>(t)) = std::sin(phase) * 0.25f + noise(rng);
        for (int channel = 1; channel < cfg.inputSize; ++channel) {
            const float offset = static_cast<float>(channel - 1) /
                                 static_cast<float>(std::max(cfg.inputSize - 1, 1));
            const float sweep = 0.5f + 0.5f * std::sin(phase * 0.25f + offset * 6.283185307f);
            input(channel, static_cast<int>(t)) = std::clamp(sweep + noise(rng), 0.0f, 1.0f);
        }
    }
    return input;
}

} // namespace

BenchmarkResult benchmarkStreamingInference(const NeuralNet& net, const BenchmarkOptions& options) {
    if (options.samples == 0) {
        throw std::runtime_error("benchmark sample count must be greater than zero");
    }

    const std::size_t totalSamples = options.samples + options.warmupSamples;
    const Eigen::MatrixXf input = makeBenchmarkInput(net.modelConfig(), totalSamples, options.seed);
    InferenceState state = net.makeInferenceState();

    for (std::size_t i = 0; i < options.warmupSamples; ++i) {
        static_cast<void>(net.predictSample(input.col(static_cast<int>(i)), state));
    }

    std::vector<double> latencies;
    latencies.reserve(options.samples);
    float checksum = 0.0f;
    const auto totalStart = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < options.samples; ++i) {
        const int column = static_cast<int>(i + options.warmupSamples);
        const auto start = std::chrono::steady_clock::now();
        checksum += net.predictSample(input.col(column), state);
        const auto stop = std::chrono::steady_clock::now();
        latencies.push_back(
            std::chrono::duration<double, std::micro>(stop - start).count());
    }
    const auto totalStop = std::chrono::steady_clock::now();

    std::sort(latencies.begin(), latencies.end());
    const double totalSeconds = std::chrono::duration<double>(totalStop - totalStart).count();
    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);

    BenchmarkResult result;
    result.samples = options.samples;
    result.meanMicroseconds = sum / static_cast<double>(options.samples);
    result.p50Microseconds = percentile(latencies, 0.50);
    result.p95Microseconds = percentile(latencies, 0.95);
    result.p99Microseconds = percentile(latencies, 0.99);
    result.maxMicroseconds = latencies.back();
    result.samplesPerSecond = static_cast<double>(options.samples) / std::max(totalSeconds, 1.0e-12);
    result.checksum = checksum;
    return result;
}

} // namespace stock_signal
