#pragma once

#include "Config.h"

#include <Eigen/Dense>

#include <filesystem>
#include <vector>

namespace neural_amp {

struct Sequence {
    Eigen::MatrixXf input;     // channels x time, where channels = audio + bands
    Eigen::RowVectorXf target; // 1 x time
};

struct Dataset {
    int sampleRate = 0;
    float durationSeconds = 0.0f;
    int chunkLength = 0;
    int hopLength = 0;
    int bandCount = 0;
    std::uint32_t seed = 0;
    std::vector<Sequence> sequences;
};

Dataset generateDataset(const Config& cfg);
void saveDataset(const std::filesystem::path& outputDir, const Dataset& dataset);
Dataset loadDataset(const std::filesystem::path& datasetPath);
std::filesystem::path resolveDatasetBin(const std::filesystem::path& datasetPath);

} // namespace neural_amp
