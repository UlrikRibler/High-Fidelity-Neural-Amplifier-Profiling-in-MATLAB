#pragma once

#include "Model.h"

#include <filesystem>
#include <string>

namespace neural_amp {

struct CheckpointData {
    NeuralNet net;
    int epoch = 0;
    std::int64_t optimizerStep = 0;
};

void saveModel(const std::filesystem::path& path, const NeuralNet& net);
NeuralNet loadModel(const std::filesystem::path& path);

void saveCheckpoint(const std::filesystem::path& path, const NeuralNet& net, int epoch,
                    std::int64_t optimizerStep);
CheckpointData loadCheckpoint(const std::filesystem::path& path);
std::string inspectCheckpoint(const std::filesystem::path& path);

} // namespace neural_amp
