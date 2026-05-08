#include "Trainer.h"

#include "Artifacts.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>

namespace stock_signal {

void computeNormalization(const Dataset& dataset, Eigen::VectorXf& mean, Eigen::VectorXf& stddev) {
    if (dataset.sequences.empty()) {
        throw std::runtime_error("cannot normalize empty dataset");
    }
    const int channels = static_cast<int>(dataset.sequences.front().input.rows());
    mean = Eigen::VectorXf::Zero(channels);
    stddev = Eigen::VectorXf::Zero(channels);
    std::int64_t count = 0;

    for (const Sequence& seq : dataset.sequences) {
        mean += seq.input.rowwise().sum();
        count += seq.input.cols();
    }
    mean /= static_cast<float>(count);

    for (const Sequence& seq : dataset.sequences) {
        const Eigen::MatrixXf centered = seq.input.colwise() - mean;
        stddev += centered.array().square().rowwise().sum().matrix();
    }
    stddev = (stddev / static_cast<float>(count)).array().sqrt().max(1.0e-6f);
}

std::filesystem::path findLatestCheckpoint(const std::filesystem::path& checkpointDir) {
    if (!std::filesystem::exists(checkpointDir)) {
        return {};
    }
    std::regex pattern(R"(checkpoint_epoch_([0-9]+)\.bin)");
    int bestEpoch = -1;
    std::filesystem::path bestPath;
    for (const auto& entry : std::filesystem::directory_iterator(checkpointDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::smatch match;
        const std::string name = entry.path().filename().string();
        if (std::regex_match(name, match, pattern)) {
            const int epoch = std::stoi(match[1].str());
            if (epoch > bestEpoch) {
                bestEpoch = epoch;
                bestPath = entry.path();
            }
        }
    }
    return bestPath;
}

TrainingResult trainModel(const Dataset& dataset, const Config& cfg, const std::filesystem::path& outputDir) {
    if (dataset.sequences.empty()) {
        throw std::runtime_error("cannot train on empty dataset");
    }

    const std::filesystem::path checkpointDir = outputDir / "checkpoints";
    std::filesystem::create_directories(checkpointDir);

    ModelConfig modelConfig;
    modelConfig.inputSize = static_cast<int>(dataset.sequences.front().input.rows());
    modelConfig.hidden1 = cfg.hidden1;
    modelConfig.hidden2 = cfg.hidden2;
    modelConfig.dense = cfg.dense;

    NeuralNet net(modelConfig, cfg.seed);
    int startEpoch = 0;
    std::int64_t optimizerStep = 0;
    const std::filesystem::path latest = findLatestCheckpoint(checkpointDir);
    if (!latest.empty()) {
        CheckpointData checkpoint = loadCheckpoint(latest);
        net = checkpoint.net;
        startEpoch = checkpoint.epoch;
        optimizerStep = checkpoint.optimizerStep;
        std::cout << "Resuming from " << latest.string() << '\n';
    } else {
        Eigen::VectorXf mean;
        Eigen::VectorXf stddev;
        computeNormalization(dataset, mean, stddev);
        net.setNormalization(mean, stddev);
    }

    std::vector<int> order(dataset.sequences.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(cfg.seed + 1U);

    for (int epoch = startEpoch + 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const int drops = cfg.learningRateDropPeriod > 0 ? (epoch - 1) / cfg.learningRateDropPeriod : 0;
        const float lr = cfg.learningRate * std::pow(cfg.learningRateDropFactor, static_cast<float>(drops));

        float epochLoss = 0.0f;
        int windows = 0;
        int accumulated = 0;
        net.zeroGradients();

        for (int sequenceIndex : order) {
            const Sequence& seq = dataset.sequences[static_cast<std::size_t>(sequenceIndex)];
            Eigen::VectorXf h1 = Eigen::VectorXf::Zero(cfg.hidden1);
            Eigen::VectorXf h2 = Eigen::VectorXf::Zero(cfg.hidden2);

            const int sequenceLength = static_cast<int>(seq.input.cols());
            for (int start = 0; start < sequenceLength; start += cfg.truncationLength) {
                const int length = std::min(cfg.truncationLength, sequenceLength - start);
                TrainWindowResult window =
                    net.accumulateGradients(seq.input, seq.target, start, length, h1, h2);
                h1 = window.h1Final;
                h2 = window.h2Final;
                epochLoss += window.loss;
                ++windows;
                ++accumulated;

                if (accumulated >= cfg.batchSize) {
                    net.scaleGradients(1.0f / static_cast<float>(accumulated));
                    net.clipGradients(cfg.gradientClipNorm);
                    net.adamStep(lr, ++optimizerStep);
                    net.zeroGradients();
                    accumulated = 0;
                }
            }
        }

        if (accumulated > 0) {
            net.scaleGradients(1.0f / static_cast<float>(accumulated));
            net.clipGradients(cfg.gradientClipNorm);
            net.adamStep(lr, ++optimizerStep);
            net.zeroGradients();
        }

        const float meanLoss = epochLoss / static_cast<float>(std::max(windows, 1));
        std::cout << "Epoch " << epoch << "/" << cfg.epochs << " | loss " << meanLoss
                  << " | lr " << lr << '\n';

        char filename[64]{};
#ifdef _WIN32
        sprintf_s(filename, "checkpoint_epoch_%04d.bin", epoch);
#else
        std::snprintf(filename, sizeof(filename), "checkpoint_epoch_%04d.bin", epoch);
#endif
        saveCheckpoint(checkpointDir / filename, net, epoch, optimizerStep);
    }

    return {net, cfg.epochs, optimizerStep};
}

ValidationResult validateModel(const NeuralNet& net, const Dataset& dataset) {
    if (dataset.sequences.empty()) {
        throw std::runtime_error("cannot validate empty dataset");
    }
    const Sequence& seq = dataset.sequences.front();
    const Eigen::RowVectorXf prediction = net.predict(seq.input);
    const Eigen::RowVectorXf error = seq.target - prediction;
    const float denom = std::max(seq.target.squaredNorm(), 1.0e-12f);
    const float esr = error.squaredNorm() / denom;
    return {esr, (1.0f - esr) * 100.0f};
}

} // namespace stock_signal
