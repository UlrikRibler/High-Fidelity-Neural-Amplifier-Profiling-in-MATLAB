#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

namespace stock_signal {

struct Config {
    std::string presetName = "quick";
    int sampleRate = 96000;
    float durationSeconds = 2.0f;
    int epochs = 2;
    int batchSize = 4;
    float chunkSeconds = 0.10f;
    float hopSeconds = 0.05f;
    int hidden1 = 8;
    int hidden2 = 4;
    int dense = 8;
    int bandCount = 20;
    float minBandHz = 0.0f;
    float maxBandHz = 20000.0f;
    int truncationLength = 128;
    float learningRate = 0.003f;
    int learningRateDropPeriod = 50;
    float learningRateDropFactor = 0.5f;
    float gradientClipNorm = 5.0f;
    std::uint32_t seed = 1337;
};

Config makePreset(const std::string& name);
nlohmann::json toJson(const Config& cfg);
Config configFromJson(const nlohmann::json& json);
void saveConfig(const std::filesystem::path& path, const Config& cfg);
Config loadConfig(const std::filesystem::path& path);
std::string timestampedSessionName();

} // namespace stock_signal
