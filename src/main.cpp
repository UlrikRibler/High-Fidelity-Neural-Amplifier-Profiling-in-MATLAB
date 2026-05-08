#include "Artifacts.h"
#include "Config.h"
#include "Pipeline.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using neural_amp::Config;

struct ParsedArgs {
    std::string command;
    std::map<std::string, std::string> options;
};

void printHelp() {
    std::cout
        << "neural_amp - C++ neural amplifier profiling\n\n"
        << "Commands:\n"
        << "  run --preset quick [--output experiments/<session>] [--duration N] [--epochs N] [--bands 20]\n"
        << "  generate --preset quick --output experiments/<session> [--bands 20]\n"
        << "  train --dataset <path> --output experiments/<session> [--preset quick]\n"
        << "  validate --model <path> --dataset <path>\n"
        << "  inspect --checkpoint <path>\n\n"
        << "Presets: quick, gen4-full\n";
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
    Config cfg = neural_amp::makePreset(option(args, "--preset", "quick"));
    if (args.options.contains("--duration")) {
        cfg.durationSeconds = std::stof(args.options.at("--duration"));
    }
    if (args.options.contains("--epochs")) {
        cfg.epochs = std::stoi(args.options.at("--epochs"));
    }
    if (args.options.contains("--bands")) {
        cfg.bandCount = std::stoi(args.options.at("--bands"));
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
    return std::filesystem::path("experiments") / neural_amp::timestampedSessionName();
}

Config configForTrain(const ParsedArgs& args, const std::filesystem::path& datasetPath,
                      const std::filesystem::path& outputDir) {
    const auto explicitPreset = args.options.find("--preset");
    if (explicitPreset != args.options.end()) {
        return configFromOptions(args);
    }
    const std::filesystem::path outputConfigPath = outputDir / "config.json";
    if (std::filesystem::exists(outputConfigPath)) {
        return neural_amp::loadConfig(outputConfigPath);
    }
    const std::filesystem::path datasetConfigPath =
        std::filesystem::is_directory(datasetPath) ? datasetPath / "config.json"
                                                   : datasetPath.parent_path() / "config.json";
    if (std::filesystem::exists(datasetConfigPath)) {
        return neural_amp::loadConfig(datasetConfigPath);
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
            neural_amp::runPipeline(cfg, defaultOutput(args));
            return 0;
        }

        if (args.command == "generate") {
            const Config cfg = configFromOptions(args);
            neural_amp::generateCommand(cfg, defaultOutput(args));
            return 0;
        }

        if (args.command == "train") {
            const std::filesystem::path output = defaultOutput(args);
            const std::filesystem::path dataset = requiredPath(args, "--dataset");
            const Config cfg = configForTrain(args, dataset, output);
            neural_amp::trainCommand(cfg, dataset, output);
            return 0;
        }

        if (args.command == "validate") {
            neural_amp::validateCommand(requiredPath(args, "--model"), requiredPath(args, "--dataset"));
            return 0;
        }

        if (args.command == "inspect") {
            std::cout << neural_amp::inspectCheckpoint(requiredPath(args, "--checkpoint")) << '\n';
            return 0;
        }

        throw std::runtime_error("unknown command: " + args.command);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        printHelp();
        return 1;
    }
}
