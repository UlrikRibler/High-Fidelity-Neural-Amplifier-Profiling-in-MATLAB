#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

namespace stock_signal {

struct CheckpointData;

struct ModelConfig {
    int inputSize = 21;
    int hidden1 = 8;
    int hidden2 = 4;
    int dense = 8;
};

struct Param {
    Eigen::MatrixXf value;
    Eigen::MatrixXf grad;
    Eigen::MatrixXf m;
    Eigen::MatrixXf v;

    Param() = default;
    Param(int rows, int cols);
    void zeroGrad();
};

class GRULayer {
public:
    GRULayer() = default;
    GRULayer(int inputSize, int hiddenSize, std::mt19937& rng);

    int inputSize() const { return inputSize_; }
    int hiddenSize() const { return hiddenSize_; }
    std::vector<Param*> parameters();
    std::vector<const Param*> parameters() const;

private:
    friend class NeuralNet;

    struct StepCache {
        Eigen::VectorXf x;
        Eigen::VectorXf hPrev;
        Eigen::VectorXf z;
        Eigen::VectorXf r;
        Eigen::VectorXf n;
        Eigen::VectorXf h;
    };

    Eigen::VectorXf forwardStep(const Eigen::VectorXf& x, const Eigen::VectorXf& hPrev,
                                StepCache& cache) const;
    void forwardInference(const Eigen::Ref<const Eigen::VectorXf>& x,
                          Eigen::VectorXf& h,
                          Eigen::VectorXf& z,
                          Eigen::VectorXf& r,
                          Eigen::VectorXf& n,
                          Eigen::VectorXf& scratch) const;
    void backwardStep(const StepCache& cache, const Eigen::VectorXf& dh,
                      Eigen::VectorXf& dx, Eigen::VectorXf& dhPrev);

    int inputSize_ = 0;
    int hiddenSize_ = 0;
    Param wz_;
    Param wr_;
    Param wn_;
    Param uz_;
    Param ur_;
    Param un_;
    Param bz_;
    Param br_;
    Param bn_;
};

class DenseLayer {
public:
    DenseLayer() = default;
    DenseLayer(int inputSize, int outputSize, std::mt19937& rng);

    std::vector<Param*> parameters();
    std::vector<const Param*> parameters() const;

private:
    friend class NeuralNet;

    Param w_;
    Param b_;
};

struct TrainWindowResult {
    float loss = 0.0f;
    Eigen::VectorXf h1Final;
    Eigen::VectorXf h2Final;
};

struct InferenceState {
    Eigen::VectorXf h1;
    Eigen::VectorXf h2;
    Eigen::VectorXf xNorm;
    Eigen::VectorXf z1;
    Eigen::VectorXf r1;
    Eigen::VectorXf n1;
    Eigen::VectorXf scratch1;
    Eigen::VectorXf z2;
    Eigen::VectorXf r2;
    Eigen::VectorXf n2;
    Eigen::VectorXf scratch2;
    Eigen::VectorXf densePre;
    Eigen::VectorXf denseAct;

    InferenceState() = default;
    explicit InferenceState(const ModelConfig& cfg);
    void resize(const ModelConfig& cfg);
    void reset();
};

class NeuralNet {
public:
    NeuralNet();
    NeuralNet(ModelConfig cfg, std::uint32_t seed);

    ModelConfig modelConfig() const { return cfg_; }
    std::vector<Param*> parameters();
    std::vector<const Param*> parameters() const;

    void setNormalization(const Eigen::VectorXf& mean, const Eigen::VectorXf& stddev);
    const Eigen::VectorXf& normalizationMean() const { return normMean_; }
    const Eigen::VectorXf& normalizationStd() const { return normStd_; }

    InferenceState makeInferenceState() const;
    float predictSample(const Eigen::Ref<const Eigen::VectorXf>& input,
                        InferenceState& state) const;
    Eigen::RowVectorXf predict(const Eigen::MatrixXf& input) const;
    TrainWindowResult accumulateGradients(const Eigen::MatrixXf& input,
                                          const Eigen::RowVectorXf& target,
                                          int start, int length,
                                          const Eigen::VectorXf& h1Initial,
                                          const Eigen::VectorXf& h2Initial);
    float lossOnly(const Eigen::MatrixXf& input, const Eigen::RowVectorXf& target,
                   int start, int length) const;

    void zeroGradients();
    void scaleGradients(float scale);
    float gradientNorm() const;
    void clipGradients(float maxNorm);
    void adamStep(float learningRate, std::int64_t step);

private:
    friend struct CheckpointData;
    friend void saveModel(const std::filesystem::path&, const NeuralNet&);
    friend NeuralNet loadModel(const std::filesystem::path&);
    friend void saveCheckpoint(const std::filesystem::path&, const NeuralNet&, int, std::int64_t);
    friend CheckpointData loadCheckpoint(const std::filesystem::path&);

    Eigen::VectorXf normalizeInputColumn(const Eigen::MatrixXf& input, int column) const;

    ModelConfig cfg_;
    GRULayer gru1_;
    GRULayer gru2_;
    DenseLayer shaper_;
    DenseLayer output_;
    Eigen::VectorXf normMean_;
    Eigen::VectorXf normStd_;
};

Eigen::VectorXf sigmoid(const Eigen::VectorXf& x);

} // namespace stock_signal
