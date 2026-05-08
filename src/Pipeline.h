#pragma once

#include "Config.h"

#include <filesystem>

namespace stock_signal {

void runPipeline(const Config& cfg, const std::filesystem::path& outputDir);
void generateCommand(const Config& cfg, const std::filesystem::path& outputDir);
void trainCommand(const Config& cfg, const std::filesystem::path& datasetPath,
                  const std::filesystem::path& outputDir);
void validateCommand(const std::filesystem::path& modelPath, const std::filesystem::path& datasetPath);
void candleSignalCommand(const Config& cfg,
                         const std::filesystem::path& candlePath,
                         const std::filesystem::path& outputDir,
                         const std::string& interval,
                         float secondsPerCandle);

} // namespace stock_signal
