#include "Pipeline.h"

#include "Artifacts.h"
#include "CandleSignal.h"
#include "Dataset.h"
#include "Trainer.h"

#include <filesystem>
#include <iostream>

namespace stock_signal {

void runPipeline(const Config& cfg, const std::filesystem::path& outputDir) {
    std::filesystem::create_directories(outputDir);
    std::cout << "Generating stock-signal dataset (" << cfg.presetName << ")...\n";
    saveConfig(outputDir / "config.json", cfg);
    Dataset dataset = generateDataset(cfg);
    saveDataset(outputDir, dataset);

    std::cout << "Training model...\n";
    TrainingResult result = trainModel(dataset, cfg, outputDir);

    std::cout << "Validating final model...\n";
    const ValidationResult validation = validateModel(result.net, dataset);
    std::cout << "Validation ESR: " << validation.esr << " | Accuracy: "
              << validation.accuracyPercent << "%\n";
    saveModel(outputDir / "final_model.bin", result.net);
    std::cout << "Final model: " << (outputDir / "final_model.bin").string() << '\n';
}

void generateCommand(const Config& cfg, const std::filesystem::path& outputDir) {
    std::filesystem::create_directories(outputDir);
    saveConfig(outputDir / "config.json", cfg);
    Dataset dataset = generateDataset(cfg);
    saveDataset(outputDir, dataset);
    std::cout << "Dataset written to " << outputDir.string() << '\n';
}

void trainCommand(const Config& cfg, const std::filesystem::path& datasetPath,
                  const std::filesystem::path& outputDir) {
    std::filesystem::create_directories(outputDir);
    saveConfig(outputDir / "config.json", cfg);
    Dataset dataset = loadDataset(datasetPath);
    TrainingResult result = trainModel(dataset, cfg, outputDir);
    saveModel(outputDir / "final_model.bin", result.net);
    std::cout << "Final model: " << (outputDir / "final_model.bin").string() << '\n';
}

void validateCommand(const std::filesystem::path& modelPath, const std::filesystem::path& datasetPath) {
    NeuralNet net = loadModel(modelPath);
    Dataset dataset = loadDataset(datasetPath);
    const ValidationResult validation = validateModel(net, dataset);
    std::cout << "Validation ESR: " << validation.esr << " | Accuracy: "
              << validation.accuracyPercent << "%\n";
}

void candleSignalCommand(const Config& cfg,
                         const std::filesystem::path& candlePath,
                         const std::filesystem::path& outputDir,
                         const std::string& interval,
                         float secondsPerCandle) {
    std::filesystem::create_directories(outputDir);
    saveConfig(outputDir / "config.json", cfg);

    CandleSignalOptions options;
    options.sampleRate = cfg.sampleRate;
    options.secondsPerCandle = secondsPerCandle;
    options.bandCount = cfg.bandCount;
    options.minBandHz = cfg.minBandHz;
    options.maxBandHz = cfg.maxBandHz;
    options.interval = interval;
    options.seed = cfg.seed;

    std::cout << "Loading chart candles from " << candlePath.string() << "...\n";
    const std::vector<Candle> candles = loadCandles(candlePath);
    std::cout << "Converting " << candles.size() << " candles at interval " << interval
              << " to " << cfg.sampleRate << " Hz signal...\n";
    const CandleSignal signal = candlesToSignal(candles, options);
    saveCandleSignalArtifacts(outputDir, signal, options);

    const Dataset dataset = datasetFromCandleSignal(signal, cfg);
    saveDataset(outputDir, dataset);
    std::cout << "Candle signal tool artifacts written to " << outputDir.string() << '\n';
    std::cout << "Input WAV: " << (outputDir / "candle_waveform.wav").string() << '\n';
    std::cout << "Response WAV: " << (outputDir / "market_signal_response.wav").string() << '\n';
}

} // namespace stock_signal
