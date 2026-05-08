#pragma once

#include "Config.h"
#include "Dataset.h"
#include "Model.h"

#include <filesystem>

namespace stock_signal {

struct TrainingResult {
    NeuralNet net;
    int epochsCompleted = 0;
    std::int64_t optimizerStep = 0;
};

struct ValidationResult {
    float esr = 0.0f;
    float accuracyPercent = 0.0f;
};

void computeNormalization(const Dataset& dataset, Eigen::VectorXf& mean, Eigen::VectorXf& stddev);
TrainingResult trainModel(const Dataset& dataset, const Config& cfg, const std::filesystem::path& outputDir);
ValidationResult validateModel(const NeuralNet& net, const Dataset& dataset);
std::filesystem::path findLatestCheckpoint(const std::filesystem::path& checkpointDir);

} // namespace stock_signal
