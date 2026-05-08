#pragma once

#include "Config.h"
#include "Dataset.h"

#include <filesystem>
#include <string>
#include <vector>

namespace stock_signal {

struct Candle {
    double time = 0.0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
};

struct CandleSignalOptions {
    int sampleRate = 192000;
    float secondsPerCandle = 0.02f;
    int bandCount = 20;
    float minBandHz = 0.0f;
    float maxBandHz = 20000.0f;
    std::string interval = "chart";
    std::uint32_t seed = 1337;
};

struct CandleSignal {
    std::vector<Candle> candles;
    std::vector<float> waveform;
    std::vector<std::vector<float>> bandControls;
    std::vector<float> processed;
    int samplesPerCandle = 0;
};

std::vector<Candle> loadCandles(const std::filesystem::path& path);
CandleSignal candlesToSignal(const std::vector<Candle>& candles, const CandleSignalOptions& options);
Dataset datasetFromCandleSignal(const CandleSignal& signal, const Config& cfg);
void saveWav(const std::filesystem::path& path, const std::vector<float>& signal, int sampleRate);
void saveCandleSignalArtifacts(const std::filesystem::path& outputDir,
                               const CandleSignal& signal,
                               const CandleSignalOptions& options);

} // namespace stock_signal
