#include "Pipeline.h"

#include "Artifacts.h"
#include "Dataset.h"
#include "Trainer.h"

#include <filesystem>
#include <iostream>

namespace neural_amp {

void runPipeline(const Config& cfg, const std::filesystem::path& outputDir) {
    std::filesystem::create_directories(outputDir);
    std::cout << "Generating dataset (" << cfg.presetName << ")...\n";
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

} // namespace neural_amp
