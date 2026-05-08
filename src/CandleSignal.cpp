#include "CandleSignal.h"

#include "Dsp.h"
#include "MarketSignalProcessor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace stock_signal {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    bool quoted = false;
    for (char c : line) {
        if (c == '"') {
            quoted = !quoted;
            continue;
        }
        if (c == ',' && !quoted) {
            out.push_back(cell);
            cell.clear();
            continue;
        }
        cell.push_back(c);
    }
    out.push_back(cell);
    return out;
}

double jsonNumber(const nlohmann::json& object, std::initializer_list<const char*> keys, double fallback = 0.0) {
    for (const char* key : keys) {
        const auto it = object.find(key);
        if (it == object.end() || it->is_null()) {
            continue;
        }
        if (it->is_number()) {
            return it->get<double>();
        }
        if (it->is_string()) {
            try {
                return std::stod(it->get<std::string>());
            } catch (...) {
                continue;
            }
        }
    }
    return fallback;
}

Candle candleFromJson(const nlohmann::json& object) {
    Candle candle;
    candle.time = jsonNumber(object, {"time", "timestamp", "t"}, 0.0);
    candle.open = jsonNumber(object, {"open", "o", "price_usd", "close", "c"}, 0.0);
    candle.high = jsonNumber(object, {"high", "h", "price_usd", "close", "c"}, candle.open);
    candle.low = jsonNumber(object, {"low", "l", "price_usd", "close", "c"}, candle.open);
    candle.close = jsonNumber(object, {"close", "c", "price_usd"}, candle.open);
    candle.volume = jsonNumber(object, {"volume", "v"}, 0.0);
    if (candle.high < candle.low) {
        std::swap(candle.high, candle.low);
    }
    return candle;
}

std::vector<Candle> loadJsonCandles(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read candles: " + path.string());
    }
    nlohmann::json json;
    in >> json;

    nlohmann::json array;
    if (json.is_array()) {
        array = json;
    } else if (json.contains("candles") && json["candles"].is_array()) {
        array = json["candles"];
    } else if (json.contains("series") && json["series"].is_array()) {
        array = json["series"];
    } else {
        throw std::runtime_error("candle JSON must be an array or contain candles/series");
    }

    std::vector<Candle> candles;
    candles.reserve(array.size());
    for (const auto& item : array) {
        if (item.is_object()) {
            Candle candle = candleFromJson(item);
            if (candle.close > 0.0) {
                candles.push_back(candle);
            }
        }
    }
    return candles;
}

double csvValue(const std::vector<std::string>& row,
                const std::map<std::string, int>& columns,
                std::initializer_list<const char*> names,
                double fallback = 0.0) {
    for (const char* name : names) {
        const auto it = columns.find(name);
        if (it == columns.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) {
            continue;
        }
        try {
            return std::stod(row[static_cast<std::size_t>(it->second)]);
        } catch (...) {
            continue;
        }
    }
    return fallback;
}

std::vector<Candle> loadCsvCandles(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read candles: " + path.string());
    }

    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        throw std::runtime_error("candle CSV is empty");
    }
    const std::vector<std::string> headers = splitCsvLine(headerLine);
    std::map<std::string, int> columns;
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        columns[lower(headers[static_cast<std::size_t>(i)])] = i;
    }

    std::vector<Candle> candles;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = splitCsvLine(line);
        Candle candle;
        candle.time = csvValue(row, columns, {"time", "timestamp", "date"}, 0.0);
        candle.open = csvValue(row, columns, {"open", "o"}, 0.0);
        candle.high = csvValue(row, columns, {"high", "h"}, candle.open);
        candle.low = csvValue(row, columns, {"low", "l"}, candle.open);
        candle.close = csvValue(row, columns, {"close", "c", "price"}, candle.open);
        candle.volume = csvValue(row, columns, {"volume", "v"}, 0.0);
        if (candle.high < candle.low) {
            std::swap(candle.high, candle.low);
        }
        if (candle.close > 0.0) {
            candles.push_back(candle);
        }
    }
    return candles;
}

float safeLogReturn(double current, double previous) {
    if (current <= 0.0 || previous <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::log(current / previous));
}

float rmsScale(const std::vector<float>& values, float fallback) {
    double sumSquares = 0.0;
    for (float value : values) {
        sumSquares += static_cast<double>(value) * static_cast<double>(value);
    }
    if (values.empty()) {
        return fallback;
    }
    return std::max(fallback, static_cast<float>(std::sqrt(sumSquares / static_cast<double>(values.size()))));
}

void writeBytes(std::ostream& out, const void* data, std::size_t size) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error("failed to write wav");
    }
}

template <typename T>
void writeScalar(std::ostream& out, T value) {
    writeBytes(out, &value, sizeof(T));
}

} // namespace

std::vector<Candle> loadCandles(const std::filesystem::path& path) {
    const std::string ext = lower(path.extension().string());
    std::vector<Candle> candles = (ext == ".json") ? loadJsonCandles(path) : loadCsvCandles(path);
    if (candles.size() < 2) {
        throw std::runtime_error("at least two valid candles are required");
    }
    return candles;
}

CandleSignal candlesToSignal(const std::vector<Candle>& candles, const CandleSignalOptions& options) {
    if (candles.size() < 2) {
        throw std::runtime_error("at least two candles are required");
    }
    if (options.sampleRate <= 0 || options.secondsPerCandle <= 0.0f || options.bandCount <= 0) {
        throw std::runtime_error("invalid candle signal options");
    }

    CandleSignal signal;
    signal.candles = candles;
    signal.samplesPerCandle =
        std::max(16, static_cast<int>(std::lround(options.secondsPerCandle * options.sampleRate)));
    const int totalSamples = signal.samplesPerCandle * static_cast<int>(candles.size());
    signal.waveform.assign(static_cast<std::size_t>(totalSamples), 0.0f);
    signal.bandControls.assign(static_cast<std::size_t>(options.bandCount),
                               std::vector<float>(static_cast<std::size_t>(totalSamples), 0.0f));

    std::vector<float> returns;
    std::vector<float> ranges;
    std::vector<float> volumes;
    returns.reserve(candles.size());
    ranges.reserve(candles.size());
    volumes.reserve(candles.size());
    for (std::size_t i = 0; i < candles.size(); ++i) {
        const Candle& candle = candles[i];
        const double prevClose = i == 0 ? candle.open : candles[i - 1].close;
        returns.push_back(safeLogReturn(candle.close, prevClose));
        ranges.push_back(static_cast<float>(
            candle.close > 0.0 ? std::max(0.0, (candle.high - candle.low) / candle.close) : 0.0));
        volumes.push_back(static_cast<float>(std::log1p(std::max(0.0, candle.volume))));
    }

    const float returnScale = rmsScale(returns, 0.0025f);
    const float rangeScale = rmsScale(ranges, 0.01f);
    const float volumeScale = rmsScale(volumes, 1.0f);

    std::mt19937 rng(options.seed);
    std::normal_distribution<float> microNoise(0.0f, 0.008f);

    for (std::size_t candleIndex = 0; candleIndex < candles.size(); ++candleIndex) {
        const Candle& candle = candles[candleIndex];
        const double prevClose = candleIndex == 0 ? candle.open : candles[candleIndex - 1].close;
        const float open = safeLogReturn(candle.open, prevClose) / returnScale;
        const float close = safeLogReturn(candle.close, prevClose) / returnScale;
        const float high = safeLogReturn(std::max(candle.high, candle.close), prevClose) / returnScale;
        const float low = safeLogReturn(std::min(candle.low, candle.close), prevClose) / returnScale;
        const float direction = close >= open ? 1.0f : -1.0f;
        const float range = std::clamp((high - low) / std::max(rangeScale / returnScale, 1.0e-4f), 0.0f, 4.0f);
        const float volume = std::clamp(static_cast<float>(std::log1p(std::max(0.0, candle.volume))) /
                                            std::max(volumeScale, 1.0e-4f),
                                        0.0f,
                                        4.0f);
        const float momentum = std::clamp(std::abs(returns[candleIndex]) / returnScale, 0.0f, 4.0f);

        const int start = static_cast<int>(candleIndex) * signal.samplesPerCandle;
        for (int s = 0; s < signal.samplesPerCandle; ++s) {
            const float p = signal.samplesPerCandle <= 1
                                ? 0.0f
                                : static_cast<float>(s) / static_cast<float>(signal.samplesPerCandle - 1);
            const float smooth = p * p * (3.0f - 2.0f * p);
            const float trend = open + (close - open) * smooth;
            const float wick = std::sin(3.14159265358979323846f * p) * range * direction;
            const float pulse = std::sin(2.0f * 3.14159265358979323846f * (1.0f + volume) * p) *
                                std::min(volume, 2.0f);
            const float value = 0.58f * trend + 0.32f * wick + 0.10f * pulse + microNoise(rng);
            const int index = start + s;
            signal.waveform[static_cast<std::size_t>(index)] = std::clamp(value, -4.0f, 4.0f);

            for (int band = 0; band < options.bandCount; ++band) {
                const float bandPos = options.bandCount <= 1
                                          ? 0.0f
                                          : static_cast<float>(band) / static_cast<float>(options.bandCount - 1);
                const float lowBandTrend = (1.0f - bandPos) * momentum;
                const float highBandActivity = bandPos * (0.65f * range + 0.35f * volume);
                const float intrabarMotion = std::abs(std::sin(3.14159265358979323846f * p));
                signal.bandControls[static_cast<std::size_t>(band)][static_cast<std::size_t>(index)] =
                    std::clamp(0.18f + 0.32f * lowBandTrend + 0.42f * highBandActivity +
                                   0.08f * intrabarMotion,
                               0.0f,
                               1.0f);
            }
        }
    }

    normalizePeak(signal.waveform);
    signal.processed = processMarketSignal(signal.waveform,
                                           signal.bandControls,
                                           options.sampleRate,
                                           options.minBandHz,
                                           options.maxBandHz);
    return signal;
}

Dataset datasetFromCandleSignal(const CandleSignal& signal, const Config& cfg) {
    if (signal.waveform.empty() || signal.bandControls.empty() || signal.processed.size() != signal.waveform.size()) {
        throw std::runtime_error("invalid candle signal dataset input");
    }
    if (static_cast<int>(signal.bandControls.size()) != cfg.bandCount) {
        throw std::runtime_error("candle signal band count does not match config");
    }

    Dataset dataset;
    dataset.sampleRate = cfg.sampleRate;
    dataset.durationSeconds = static_cast<float>(signal.waveform.size()) / static_cast<float>(cfg.sampleRate);
    dataset.chunkLength = static_cast<int>(std::lround(cfg.chunkSeconds * cfg.sampleRate));
    dataset.hopLength = static_cast<int>(std::lround(cfg.hopSeconds * cfg.sampleRate));
    dataset.bandCount = cfg.bandCount;
    dataset.seed = cfg.seed;
    if (dataset.chunkLength <= 0 || dataset.hopLength <= 0 || dataset.hopLength > dataset.chunkLength) {
        throw std::runtime_error("invalid chunk/hop configuration");
    }
    if (static_cast<int>(signal.waveform.size()) < dataset.chunkLength) {
        throw std::runtime_error("candle signal is shorter than the configured chunk length");
    }

    const int totalSamples = static_cast<int>(signal.waveform.size());
    const int numChunks = ((totalSamples - dataset.chunkLength) / dataset.hopLength) + 1;
    dataset.sequences.reserve(static_cast<std::size_t>(numChunks));
    for (int chunk = 0; chunk < numChunks; ++chunk) {
        const int start = chunk * dataset.hopLength;
        Sequence seq;
        seq.input.resize(1 + dataset.bandCount, dataset.chunkLength);
        seq.target.resize(dataset.chunkLength);
        for (int t = 0; t < dataset.chunkLength; ++t) {
            const std::size_t idx = static_cast<std::size_t>(start + t);
            seq.input(0, t) = signal.waveform[idx];
            for (int band = 0; band < dataset.bandCount; ++band) {
                seq.input(1 + band, t) = signal.bandControls[static_cast<std::size_t>(band)][idx];
            }
            seq.target(t) = signal.processed[idx];
        }
        dataset.sequences.push_back(std::move(seq));
    }
    return dataset;
}

void saveWav(const std::filesystem::path& path, const std::vector<float>& input, int sampleRate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write wav: " + path.string());
    }
    const std::uint16_t channels = 1;
    const std::uint16_t bitsPerSample = 16;
    const std::uint32_t byteRate =
        static_cast<std::uint32_t>(sampleRate) * channels * bitsPerSample / 8U;
    const std::uint16_t blockAlign = channels * bitsPerSample / 8U;
    const std::uint32_t dataBytes = static_cast<std::uint32_t>(input.size() * sizeof(std::int16_t));
    const std::uint32_t riffBytes = 36U + dataBytes;

    writeBytes(out, "RIFF", 4);
    writeScalar(out, riffBytes);
    writeBytes(out, "WAVE", 4);
    writeBytes(out, "fmt ", 4);
    writeScalar(out, static_cast<std::uint32_t>(16));
    writeScalar(out, static_cast<std::uint16_t>(1));
    writeScalar(out, channels);
    writeScalar(out, static_cast<std::uint32_t>(sampleRate));
    writeScalar(out, byteRate);
    writeScalar(out, blockAlign);
    writeScalar(out, bitsPerSample);
    writeBytes(out, "data", 4);
    writeScalar(out, dataBytes);

    for (float sample : input) {
        const float clipped = std::clamp(sample, -1.0f, 1.0f);
        const auto pcm = static_cast<std::int16_t>(std::lround(clipped * 32767.0f));
        writeScalar(out, pcm);
    }
}

void saveCandleSignalArtifacts(const std::filesystem::path& outputDir,
                               const CandleSignal& signal,
                               const CandleSignalOptions& options) {
    std::filesystem::create_directories(outputDir);
    saveWav(outputDir / "candle_waveform.wav", signal.waveform, options.sampleRate);
    saveWav(outputDir / "market_signal_response.wav", signal.processed, options.sampleRate);

    nlohmann::json meta = {
        {"tool", "Candle Sound Analyzer"},
        {"interval", options.interval},
        {"sample_rate", options.sampleRate},
        {"seconds_per_candle", options.secondsPerCandle},
        {"samples_per_candle", signal.samplesPerCandle},
        {"candle_count", signal.candles.size()},
        {"samples", signal.waveform.size()},
        {"band_count", options.bandCount},
        {"min_band_hz", options.minBandHz},
        {"max_band_hz", options.maxBandHz},
        {"input_wav", "candle_waveform.wav"},
        {"response_wav", "market_signal_response.wav"},
    };
    std::ofstream out(outputDir / "candle_signal.json");
    out << std::setw(2) << meta << '\n';
}

} // namespace stock_signal
