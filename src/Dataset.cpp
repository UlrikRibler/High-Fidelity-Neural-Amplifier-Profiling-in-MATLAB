#include "Dataset.h"

#include "Dsp.h"
#include "VirtualTubeAmp.h"

#include <algorithm>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace neural_amp {
namespace {

constexpr char kDatasetMagic[8] = {'N', 'A', 'D', 'A', 'T', 'A', '0', '2'};
constexpr int kVersion = 2;

template <typename T>
void writeScalar(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("failed to write dataset");
    }
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read dataset");
    }
    return value;
}

void writeBytes(std::ostream& out, const void* data, std::size_t size) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error("failed to write dataset bytes");
    }
}

void readBytes(std::istream& in, void* data, std::size_t size) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read dataset bytes");
    }
}

} // namespace

Dataset generateDataset(const Config& cfg) {
    Dataset dataset;
    dataset.sampleRate = cfg.sampleRate;
    dataset.durationSeconds = cfg.durationSeconds;
    dataset.chunkLength = static_cast<int>(std::lround(cfg.chunkSeconds * cfg.sampleRate));
    dataset.hopLength = static_cast<int>(std::lround(cfg.hopSeconds * cfg.sampleRate));
    dataset.bandCount = cfg.bandCount;
    dataset.seed = cfg.seed;

    if (dataset.chunkLength <= 0 || dataset.hopLength <= 0 || dataset.hopLength > dataset.chunkLength ||
        dataset.bandCount <= 0) {
        throw std::runtime_error("invalid chunk/hop configuration");
    }

    const int totalSamples = static_cast<int>(std::lround(cfg.durationSeconds * cfg.sampleRate));
    if (totalSamples < dataset.chunkLength) {
        throw std::runtime_error("duration is too short for the configured chunk length");
    }

    std::mt19937 rng(cfg.seed);
    std::vector<float> audio = logChirp(totalSamples, cfg.sampleRate, cfg.durationSeconds);
    std::vector<float> noise = pinkNoise(totalSamples, rng);
    for (int i = 0; i < totalSamples; ++i) {
        audio[static_cast<std::size_t>(i)] = 0.5f * audio[static_cast<std::size_t>(i)] +
                                             0.5f * noise[static_cast<std::size_t>(i)];
    }
    normalizePeak(audio);

    std::vector<std::vector<float>> bandControls(static_cast<std::size_t>(dataset.bandCount));
    const float sweepCycles = std::max(1.0f, cfg.durationSeconds / 10.0f);
    for (int band = 0; band < dataset.bandCount; ++band) {
        std::vector<float> dither =
            randomKnob(totalSamples, static_cast<float>(cfg.sampleRate), 0.15f + 0.02f * static_cast<float>(band % 8), rng);
        bandControls[static_cast<std::size_t>(band)].resize(static_cast<std::size_t>(totalSamples));
        for (int i = 0; i < totalSamples; ++i) {
            const float progress =
                totalSamples <= 1 ? 0.0f : static_cast<float>(i) / static_cast<float>(totalSamples - 1);
            const float cyclePosition =
                std::fmod(progress * sweepCycles, 1.0f) * static_cast<float>(dataset.bandCount - 1);
            const float distance = std::abs(cyclePosition - static_cast<float>(band));
            const float sweep = std::max(0.0f, 1.0f - distance);
            bandControls[static_cast<std::size_t>(band)][static_cast<std::size_t>(i)] =
                std::clamp(0.85f * sweep + 0.15f * dither[static_cast<std::size_t>(i)], 0.0f, 1.0f);
        }
    }
    std::vector<float> target =
        processVirtualTubeAmp(audio, bandControls, cfg.sampleRate, cfg.minBandHz, cfg.maxBandHz);

    const int numChunks = ((totalSamples - dataset.chunkLength) / dataset.hopLength) + 1;
    dataset.sequences.reserve(static_cast<std::size_t>(numChunks));

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        const int start = chunk * dataset.hopLength;
        Sequence seq;
        seq.input.resize(1 + dataset.bandCount, dataset.chunkLength);
        seq.target.resize(dataset.chunkLength);
        for (int t = 0; t < dataset.chunkLength; ++t) {
            const std::size_t idx = static_cast<std::size_t>(start + t);
            seq.input(0, t) = audio[idx];
            for (int band = 0; band < dataset.bandCount; ++band) {
                seq.input(1 + band, t) = bandControls[static_cast<std::size_t>(band)][idx];
            }
            seq.target(t) = target[idx];
        }
        dataset.sequences.push_back(std::move(seq));
    }

    return dataset;
}

void saveDataset(const std::filesystem::path& outputDir, const Dataset& dataset) {
    std::filesystem::create_directories(outputDir);
    const auto binPath = outputDir / "dataset.bin";
    std::ofstream out(binPath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write dataset: " + binPath.string());
    }

    writeBytes(out, kDatasetMagic, sizeof(kDatasetMagic));
    writeScalar(out, kVersion);
    writeScalar(out, static_cast<std::int32_t>(dataset.sampleRate));
    writeScalar(out, dataset.durationSeconds);
    writeScalar(out, static_cast<std::int32_t>(dataset.chunkLength));
    writeScalar(out, static_cast<std::int32_t>(dataset.hopLength));
    writeScalar(out, static_cast<std::int32_t>(dataset.bandCount));
    writeScalar(out, dataset.seed);
    writeScalar(out, static_cast<std::int32_t>(dataset.sequences.size()));
    for (const Sequence& seq : dataset.sequences) {
        writeScalar(out, static_cast<std::int32_t>(seq.input.rows()));
        writeScalar(out, static_cast<std::int32_t>(seq.input.cols()));
        writeBytes(out, seq.input.data(), sizeof(float) * static_cast<std::size_t>(seq.input.size()));
        writeScalar(out, static_cast<std::int32_t>(seq.target.size()));
        writeBytes(out, seq.target.data(), sizeof(float) * static_cast<std::size_t>(seq.target.size()));
    }

    nlohmann::json meta = {
        {"sample_rate", dataset.sampleRate},
        {"duration_seconds", dataset.durationSeconds},
        {"chunk_length", dataset.chunkLength},
        {"hop_length", dataset.hopLength},
        {"band_count", dataset.bandCount},
        {"seed", dataset.seed},
        {"sequences", dataset.sequences.size()},
        {"input_channels", 1 + dataset.bandCount},
    };
    std::ofstream metaOut(outputDir / "dataset.json");
    metaOut << std::setw(2) << meta << '\n';
}

std::filesystem::path resolveDatasetBin(const std::filesystem::path& datasetPath) {
    if (std::filesystem::is_directory(datasetPath)) {
        return datasetPath / "dataset.bin";
    }
    return datasetPath;
}

Dataset loadDataset(const std::filesystem::path& datasetPath) {
    const auto binPath = resolveDatasetBin(datasetPath);
    std::ifstream in(binPath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read dataset: " + binPath.string());
    }

    char magic[8]{};
    readBytes(in, magic, sizeof(magic));
    for (int i = 0; i < 8; ++i) {
        if (magic[i] != kDatasetMagic[i]) {
            throw std::runtime_error("dataset magic header does not match");
        }
    }

    const int version = readScalar<int>(in);
    if (version != kVersion) {
        throw std::runtime_error("unsupported dataset version");
    }

    Dataset dataset;
    dataset.sampleRate = readScalar<std::int32_t>(in);
    dataset.durationSeconds = readScalar<float>(in);
    dataset.chunkLength = readScalar<std::int32_t>(in);
    dataset.hopLength = readScalar<std::int32_t>(in);
    dataset.bandCount = readScalar<std::int32_t>(in);
    dataset.seed = readScalar<std::uint32_t>(in);
    const auto sequenceCount = readScalar<std::int32_t>(in);
    if (sequenceCount <= 0) {
        throw std::runtime_error("dataset contains no sequences");
    }
    dataset.sequences.reserve(static_cast<std::size_t>(sequenceCount));

    for (int i = 0; i < sequenceCount; ++i) {
        const auto rows = readScalar<std::int32_t>(in);
        const auto cols = readScalar<std::int32_t>(in);
        if (rows != 1 + dataset.bandCount || cols <= 0) {
            throw std::runtime_error("invalid sequence shape in dataset");
        }
        Sequence seq;
        seq.input.resize(rows, cols);
        readBytes(in, seq.input.data(), sizeof(float) * static_cast<std::size_t>(seq.input.size()));
        const auto targetLength = readScalar<std::int32_t>(in);
        if (targetLength != cols) {
            throw std::runtime_error("target length does not match input length");
        }
        seq.target.resize(targetLength);
        readBytes(in, seq.target.data(), sizeof(float) * static_cast<std::size_t>(seq.target.size()));
        dataset.sequences.push_back(std::move(seq));
    }

    return dataset;
}

} // namespace neural_amp
