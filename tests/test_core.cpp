#include "Artifacts.h"
#include "Dataset.h"
#include "Dsp.h"
#include "Trainer.h"
#include "VirtualTubeAmp.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace neural_amp;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testDspFilterStability() {
    Biquad lp = Biquad::lowpass(48000.0f, 5000.0f);
    std::vector<float> out(1024);
    for (float& sample : out) {
        sample = lp.process(1.0f);
    }
    require(out.size() == 1024, "filter changed output length");
    require(allFinite(out), "filter produced non-finite values");
}

void testKnobRange() {
    std::mt19937 rng(42);
    const std::vector<float> knob = randomKnob(2048, 48000.0f, 0.5f, rng);
    for (float value : knob) {
        require(value >= -1.0e-6f && value <= 1.0f + 1.0e-6f, "knob outside [0,1]");
    }
}

void testDatasetShape() {
    Config cfg = makePreset("quick");
    cfg.durationSeconds = 0.25f;
    Dataset dataset = generateDataset(cfg);
    require(!dataset.sequences.empty(), "dataset has no sequences");
    require(dataset.sequences.front().input.rows() == 21, "dataset input should have audio plus 20 band controls");
    require(dataset.bandCount == 20, "dataset should use 20 sweep bands by default");
    require(dataset.sequences.front().target.size() == dataset.sequences.front().input.cols(),
            "target length mismatch");
}

void testVirtualAmpFiniteNormalized() {
    const int n = 4096;
    std::vector<float> input(n, 0.1f);
    std::vector<std::vector<float>> bands(20, std::vector<float>(n, 0.5f));
    const std::vector<float> out = processVirtualTubeAmp(input, bands, 48000);
    require(out.size() == input.size(), "amp changed output length");
    require(allFinite(out), "amp produced non-finite output");
    float peak = 0.0f;
    for (float value : out) {
        peak = std::max(peak, std::abs(value));
    }
    require(peak <= 1.0001f, "amp output not normalized");
}

void testGruForwardDimensions() {
    ModelConfig cfg;
    cfg.inputSize = 21;
    cfg.hidden1 = 3;
    cfg.hidden2 = 2;
    cfg.dense = 4;
    NeuralNet net(cfg, 7);
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(21, 6);
    const Eigen::RowVectorXf output = net.predict(input);
    require(output.size() == 6, "prediction length mismatch");
}

void testGradientSanity() {
    ModelConfig cfg;
    cfg.inputSize = 21;
    cfg.hidden1 = 2;
    cfg.hidden2 = 2;
    cfg.dense = 2;
    NeuralNet net(cfg, 9);
    net.setNormalization(Eigen::VectorXf::Zero(21), Eigen::VectorXf::Ones(21));
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(21, 4) * 0.1f;
    Eigen::RowVectorXf target = Eigen::RowVectorXf::Random(4) * 0.1f;

    net.zeroGradients();
    net.accumulateGradients(input, target, 0, 4, Eigen::VectorXf::Zero(2), Eigen::VectorXf::Zero(2));
    Param* firstParam = net.parameters().front();
    const float analytic = firstParam->grad(0, 0);
    const float original = firstParam->value(0, 0);
    constexpr float eps = 1.0e-3f;

    firstParam->value(0, 0) = original + eps;
    const float plus = net.lossOnly(input, target, 0, 4);
    firstParam->value(0, 0) = original - eps;
    const float minus = net.lossOnly(input, target, 0, 4);
    firstParam->value(0, 0) = original;

    const float numeric = (plus - minus) / (2.0f * eps);
    require(std::abs(analytic - numeric) < 5.0e-3f, "GRU gradient finite-difference check failed");
}

void testCheckpointRoundTrip() {
    ModelConfig cfg;
    cfg.inputSize = 21;
    cfg.hidden1 = 3;
    cfg.hidden2 = 2;
    cfg.dense = 3;
    NeuralNet net(cfg, 11);
    net.setNormalization(Eigen::VectorXf::Constant(21, 0.25f), Eigen::VectorXf::Constant(21, 2.0f));
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "neural_amp_tests";
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "checkpoint_epoch_0001.bin";
    saveCheckpoint(path, net, 1, 3);
    const CheckpointData loaded = loadCheckpoint(path);
    require(loaded.epoch == 1, "checkpoint epoch mismatch");
    require(loaded.optimizerStep == 3, "checkpoint optimizer step mismatch");
    require(loaded.net.modelConfig().hidden1 == 3, "checkpoint model config mismatch");
}

} // namespace

int main() {
    testDspFilterStability();
    testKnobRange();
    testDatasetShape();
    testVirtualAmpFiniteNormalized();
    testGruForwardDimensions();
    testGradientSanity();
    testCheckpointRoundTrip();
    std::cout << "All neural_amp core tests passed.\n";
    return 0;
}
