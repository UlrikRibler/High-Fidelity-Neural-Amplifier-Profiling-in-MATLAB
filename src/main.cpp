#include "Artifacts.h"
#include "Benchmark.h"
#include "Config.h"
#include "Pipeline.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using stock_signal::Config;

struct ParsedArgs {
    std::string command;
    std::map<std::string, std::string> options;
};

void printHelp() {
    std::cout
        << "stock_signal - C++ stock signal DSP analyzer\n\n"
        << "Commands:\n"
        << "  run --preset quick [--output experiments/<session>] [--duration N] [--epochs N] [--bands 20]\n"
        << "  generate --preset quick --output experiments/<session> [--bands 20]\n"
        << "  candles --candles chart.csv|json --output experiments/<session> [--interval 1D] [--preset candle-analysis]\n"
        << "  train --dataset <path> --output experiments/<session> [--preset quick]\n"
        << "  validate --model <path> --dataset <path>\n"
        << "  inspect --checkpoint <path>\n\n"
        << "  benchmark --model <path> [--samples 10000] [--warmup 1000]\n\n"
        << "Candle tool options: --sample-rate 192000 --seconds-per-candle 0.02 --chunk-seconds 0.25 --min-hz 0 --max-hz 20000\n"
        << "Presets: quick, candle-analysis, gen4-full\n";
}

ParsedArgs parseArgs(int argc, char** argv) {
    ParsedArgs parsed;
    if (argc < 2) {
        parsed.command = "help";
        return parsed;
    }
    parsed.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("unexpected positional argument: " + key);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for option: " + key);
        }
        parsed.options[key] = argv[++i];
    }
    return parsed;
}

std::string option(const ParsedArgs& args, const std::string& key, const std::string& fallback) {
    const auto it = args.options.find(key);
    return it == args.options.end() ? fallback : it->second;
}

std::filesystem::path requiredPath(const ParsedArgs& args, const std::string& key) {
    const auto it = args.options.find(key);
    if (it == args.options.end()) {
        throw std::runtime_error("missing required option: " + key);
    }
    return it->second;
}

Config configFromOptions(const ParsedArgs& args) {
    Config cfg = stock_signal::makePreset(option(args, "--preset", "quick"));
    if (args.options.contains("--duration")) {
        cfg.durationSeconds = std::stof(args.options.at("--duration"));
    }
    if (args.options.contains("--epochs")) {
        cfg.epochs = std::stoi(args.options.at("--epochs"));
    }
    if (args.options.contains("--chunk-seconds")) {
        cfg.chunkSeconds = std::stof(args.options.at("--chunk-seconds"));
    }
    if (args.options.contains("--hop-seconds")) {
        cfg.hopSeconds = std::stof(args.options.at("--hop-seconds"));
    }
    if (args.options.contains("--bands")) {
        cfg.bandCount = std::stoi(args.options.at("--bands"));
    }
    if (args.options.contains("--sample-rate")) {
        cfg.sampleRate = std::stoi(args.options.at("--sample-rate"));
    }
    if (args.options.contains("--min-hz")) {
        cfg.minBandHz = std::stof(args.options.at("--min-hz"));
    }
    if (args.options.contains("--max-hz")) {
        cfg.maxBandHz = std::stof(args.options.at("--max-hz"));
    }
    if (args.options.contains("--seed")) {
        cfg.seed = static_cast<std::uint32_t>(std::stoul(args.options.at("--seed")));
    }
    return cfg;
}

std::filesystem::path defaultOutput(const ParsedArgs& args) {
    const auto explicitOutput = args.options.find("--output");
    if (explicitOutput != args.options.end()) {
        return explicitOutput->second;
    }
    return std::filesystem::path("experiments") / stock_signal::timestampedSessionName();
}

std::size_t sizeOption(const ParsedArgs& args, const std::string& key, std::size_t fallback) {
    const auto it = args.options.find(key);
    if (it == args.options.end()) {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(it->second));
}

Config configForTrain(const ParsedArgs& args, const std::filesystem::path& datasetPath,
                      const std::filesystem::path& outputDir) {
    const auto explicitPreset = args.options.find("--preset");
    if (explicitPreset != args.options.end()) {
        return configFromOptions(args);
    }
    const std::filesystem::path outputConfigPath = outputDir / "config.json";
    if (std::filesystem::exists(outputConfigPath)) {
        return stock_signal::loadConfig(outputConfigPath);
    }
    const std::filesystem::path datasetConfigPath =
        std::filesystem::is_directory(datasetPath) ? datasetPath / "config.json"
                                                   : datasetPath.parent_path() / "config.json";
    if (std::filesystem::exists(datasetConfigPath)) {
        return stock_signal::loadConfig(datasetConfigPath);
    }
    return configFromOptions(args);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ParsedArgs args = parseArgs(argc, argv);
        if (args.command == "help" || args.command == "--help" || args.command == "-h") {
            printHelp();
            return 0;
        }

        if (args.command == "run") {
            const Config cfg = configFromOptions(args);
            stock_signal::runPipeline(cfg, defaultOutput(args));
            return 0;
        }

        if (args.command == "generate") {
            const Config cfg = configFromOptions(args);
            stock_signal::generateCommand(cfg, defaultOutput(args));
            return 0;
        }

        if (args.command == "candles") {
            Config cfg = configFromOptions(args);
            const std::filesystem::path candles = requiredPath(args, "--candles");
            const std::string interval = option(args, "--interval", "chart");
            const float secondsPerCandle =
                args.options.contains("--seconds-per-candle")
                    ? std::stof(args.options.at("--seconds-per-candle"))
                    : 0.02f;
            stock_signal::candleSignalCommand(cfg, candles, defaultOutput(args), interval, secondsPerCandle);
            return 0;
        }

        if (args.command == "train") {
            const std::filesystem::path output = defaultOutput(args);
            const std::filesystem::path dataset = requiredPath(args, "--dataset");
            const Config cfg = configForTrain(args, dataset, output);
            stock_signal::trainCommand(cfg, dataset, output);
            return 0;
        }

        if (args.command == "validate") {
            stock_signal::validateCommand(requiredPath(args, "--model"), requiredPath(args, "--dataset"));
            return 0;
        }

        if (args.command == "inspect") {
            std::cout << stock_signal::inspectCheckpoint(requiredPath(args, "--checkpoint")) << '\n';
            return 0;
        }

        if (args.command == "benchmark") {
            stock_signal::BenchmarkOptions options;
            options.samples = sizeOption(args, "--samples", options.samples);
            options.warmupSamples = sizeOption(args, "--warmup", options.warmupSamples);
            if (args.options.contains("--seed")) {
                options.seed = static_cast<std::uint32_t>(std::stoul(args.options.at("--seed")));
            }

            const stock_signal::NeuralNet net = stock_signal::loadModel(requiredPath(args, "--model"));
            const stock_signal::BenchmarkResult result =
                stock_signal::benchmarkStreamingInference(net, options);
            std::cout << "Streaming inference latency (" << result.samples << " samples)\n";
            std::cout << "mean: " << result.meanMicroseconds << " us\n";
            std::cout << "p50 : " << result.p50Microseconds << " us\n";
            std::cout << "p95 : " << result.p95Microseconds << " us\n";
            std::cout << "p99 : " << result.p99Microseconds << " us\n";
            std::cout << "max : " << result.maxMicroseconds << " us\n";
            std::cout << "throughput: " << result.samplesPerSecond << " samples/s\n";
            std::cout << "checksum: " << result.checksum << '\n';
            return 0;
        }

        throw std::runtime_error("unknown command: " + args.command);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        printHelp();
        return 1;
    }
}
