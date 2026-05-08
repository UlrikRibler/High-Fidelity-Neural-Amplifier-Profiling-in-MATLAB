#include "Config.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace neural_amp {

Config makePreset(const std::string& name) {
    if (name == "quick") {
        Config cfg;
        cfg.presetName = "quick";
        return cfg;
    }
    if (name == "gen4-full") {
        Config cfg;
        cfg.presetName = "gen4-full";
        cfg.sampleRate = 192000;
        cfg.durationSeconds = 180.0f;
        cfg.epochs = 300;
        cfg.batchSize = 64;
        cfg.chunkSeconds = 0.5f;
        cfg.hopSeconds = 0.25f;
        cfg.hidden1 = 128;
        cfg.hidden2 = 64;
        cfg.dense = 32;
        cfg.bandCount = 20;
        cfg.minBandHz = 40.0f;
        cfg.maxBandHz = 20000.0f;
        cfg.truncationLength = 1024;
        cfg.learningRate = 0.005f;
        return cfg;
    }
    throw std::runtime_error("unknown preset: " + name);
}

nlohmann::json toJson(const Config& cfg) {
    return {
        {"preset", cfg.presetName},
        {"sample_rate", cfg.sampleRate},
        {"duration_seconds", cfg.durationSeconds},
        {"epochs", cfg.epochs},
        {"batch_size", cfg.batchSize},
        {"chunk_seconds", cfg.chunkSeconds},
        {"hop_seconds", cfg.hopSeconds},
        {"hidden1", cfg.hidden1},
        {"hidden2", cfg.hidden2},
        {"dense", cfg.dense},
        {"band_count", cfg.bandCount},
        {"min_band_hz", cfg.minBandHz},
        {"max_band_hz", cfg.maxBandHz},
        {"truncation_length", cfg.truncationLength},
        {"learning_rate", cfg.learningRate},
        {"learning_rate_drop_period", cfg.learningRateDropPeriod},
        {"learning_rate_drop_factor", cfg.learningRateDropFactor},
        {"gradient_clip_norm", cfg.gradientClipNorm},
        {"seed", cfg.seed},
    };
}

Config configFromJson(const nlohmann::json& json) {
    Config cfg = makePreset(json.value("preset", "quick"));
    cfg.sampleRate = json.value("sample_rate", cfg.sampleRate);
    cfg.durationSeconds = json.value("duration_seconds", cfg.durationSeconds);
    cfg.epochs = json.value("epochs", cfg.epochs);
    cfg.batchSize = json.value("batch_size", cfg.batchSize);
    cfg.chunkSeconds = json.value("chunk_seconds", cfg.chunkSeconds);
    cfg.hopSeconds = json.value("hop_seconds", cfg.hopSeconds);
    cfg.hidden1 = json.value("hidden1", cfg.hidden1);
    cfg.hidden2 = json.value("hidden2", cfg.hidden2);
    cfg.dense = json.value("dense", cfg.dense);
    cfg.bandCount = json.value("band_count", cfg.bandCount);
    cfg.minBandHz = json.value("min_band_hz", cfg.minBandHz);
    cfg.maxBandHz = json.value("max_band_hz", cfg.maxBandHz);
    cfg.truncationLength = json.value("truncation_length", cfg.truncationLength);
    cfg.learningRate = json.value("learning_rate", cfg.learningRate);
    cfg.learningRateDropPeriod = json.value("learning_rate_drop_period", cfg.learningRateDropPeriod);
    cfg.learningRateDropFactor = json.value("learning_rate_drop_factor", cfg.learningRateDropFactor);
    cfg.gradientClipNorm = json.value("gradient_clip_norm", cfg.gradientClipNorm);
    cfg.seed = json.value("seed", cfg.seed);
    return cfg;
}

void saveConfig(const std::filesystem::path& path, const Config& cfg) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write config: " + path.string());
    }
    out << std::setw(2) << toJson(cfg) << '\n';
}

Config loadConfig(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read config: " + path.string());
    }
    nlohmann::json json;
    in >> json;
    return configFromJson(json);
}

std::string timestampedSessionName() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << "Session_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << "_CPP";
    return out.str();
}

} // namespace neural_amp
