#pragma once

#include "Config.h"

#include <filesystem>

namespace neural_amp {

void runPipeline(const Config& cfg, const std::filesystem::path& outputDir);
void generateCommand(const Config& cfg, const std::filesystem::path& outputDir);
void trainCommand(const Config& cfg, const std::filesystem::path& datasetPath,
                  const std::filesystem::path& outputDir);
void validateCommand(const std::filesystem::path& modelPath, const std::filesystem::path& datasetPath);

} // namespace neural_amp
