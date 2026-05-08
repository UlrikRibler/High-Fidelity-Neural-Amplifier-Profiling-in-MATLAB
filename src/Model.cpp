#include "Model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neural_amp {
namespace {

float xavierLimit(int fanIn, int fanOut) {
    return std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
}

void initParam(Param& param, int rows, int cols, std::mt19937& rng, float limit) {
    std::uniform_real_distribution<float> dist(-limit, limit);
    param = Param(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            param.value(i, j) = dist(rng);
        }
    }
}

Eigen::VectorXf elu(const Eigen::VectorXf& x) {
    return (x.array() > 0.0f).select(x.array(), x.array().exp() - 1.0f);
}

Eigen::VectorXf eluDerivative(const Eigen::VectorXf& x) {
    return (x.array() > 0.0f).select(1.0f, x.array().exp());
}

void appendParams(std::vector<Param*>& out, std::vector<Param*> params) {
    out.insert(out.end(), params.begin(), params.end());
}

void appendParams(std::vector<const Param*>& out, std::vector<const Param*> params) {
    out.insert(out.end(), params.begin(), params.end());
}

} // namespace

Param::Param(int rows, int cols)
    : value(Eigen::MatrixXf::Zero(rows, cols)),
      grad(Eigen::MatrixXf::Zero(rows, cols)),
      m(Eigen::MatrixXf::Zero(rows, cols)),
      v(Eigen::MatrixXf::Zero(rows, cols)) {}

void Param::zeroGrad() {
    grad.setZero();
}

Eigen::VectorXf sigmoid(const Eigen::VectorXf& x) {
    return 1.0f / (1.0f + (-x.array()).exp());
}

GRULayer::GRULayer(int inputSize, int hiddenSize, std::mt19937& rng)
    : inputSize_(inputSize), hiddenSize_(hiddenSize) {
    const float inputLimit = xavierLimit(inputSize, hiddenSize);
    const float recurrentLimit = xavierLimit(hiddenSize, hiddenSize);
    initParam(wz_, hiddenSize, inputSize, rng, inputLimit);
    initParam(wr_, hiddenSize, inputSize, rng, inputLimit);
    initParam(wn_, hiddenSize, inputSize, rng, inputLimit);
    initParam(uz_, hiddenSize, hiddenSize, rng, recurrentLimit);
    initParam(ur_, hiddenSize, hiddenSize, rng, recurrentLimit);
    initParam(un_, hiddenSize, hiddenSize, rng, recurrentLimit);
    bz_ = Param(hiddenSize, 1);
    br_ = Param(hiddenSize, 1);
    bn_ = Param(hiddenSize, 1);
}

std::vector<Param*> GRULayer::parameters() {
    return {&wz_, &wr_, &wn_, &uz_, &ur_, &un_, &bz_, &br_, &bn_};
}

std::vector<const Param*> GRULayer::parameters() const {
    return {&wz_, &wr_, &wn_, &uz_, &ur_, &un_, &bz_, &br_, &bn_};
}

Eigen::VectorXf GRULayer::forwardStep(const Eigen::VectorXf& x, const Eigen::VectorXf& hPrev,
                                      StepCache& cache) const {
    cache.x = x;
    cache.hPrev = hPrev;
    cache.z = sigmoid(wz_.value * x + uz_.value * hPrev + bz_.value.col(0));
    cache.r = sigmoid(wr_.value * x + ur_.value * hPrev + br_.value.col(0));
    cache.n = (wn_.value * x + un_.value * (cache.r.array() * hPrev.array()).matrix() +
               bn_.value.col(0))
                  .array()
                  .tanh();
    cache.h = ((1.0f - cache.z.array()) * cache.n.array() +
               cache.z.array() * hPrev.array())
                  .matrix();
    return cache.h;
}

void GRULayer::backwardStep(const StepCache& cache, const Eigen::VectorXf& dh,
                            Eigen::VectorXf& dx, Eigen::VectorXf& dhPrev) {
    const Eigen::VectorXf dn = (dh.array() * (1.0f - cache.z.array())).matrix();
    const Eigen::VectorXf dz = (dh.array() * (cache.hPrev.array() - cache.n.array())).matrix();
    Eigen::VectorXf dhPrevTotal = (dh.array() * cache.z.array()).matrix();

    const Eigen::VectorXf dan = (dn.array() * (1.0f - cache.n.array().square())).matrix();
    wn_.grad.noalias() += dan * cache.x.transpose();
    un_.grad.noalias() += dan * (cache.r.array() * cache.hPrev.array()).matrix().transpose();
    bn_.grad.col(0).noalias() += dan;
    dx.noalias() += wn_.value.transpose() * dan;
    const Eigen::VectorXf dRhPrev = un_.value.transpose() * dan;
    const Eigen::VectorXf drFromN = (dRhPrev.array() * cache.hPrev.array()).matrix();
    dhPrevTotal.noalias() += (dRhPrev.array() * cache.r.array()).matrix();

    const Eigen::VectorXf dar = (drFromN.array() * cache.r.array() *
                                 (1.0f - cache.r.array()))
                                    .matrix();
    wr_.grad.noalias() += dar * cache.x.transpose();
    ur_.grad.noalias() += dar * cache.hPrev.transpose();
    br_.grad.col(0).noalias() += dar;
    dx.noalias() += wr_.value.transpose() * dar;
    dhPrevTotal.noalias() += ur_.value.transpose() * dar;

    const Eigen::VectorXf daz = (dz.array() * cache.z.array() *
                                 (1.0f - cache.z.array()))
                                    .matrix();
    wz_.grad.noalias() += daz * cache.x.transpose();
    uz_.grad.noalias() += daz * cache.hPrev.transpose();
    bz_.grad.col(0).noalias() += daz;
    dx.noalias() += wz_.value.transpose() * daz;
    dhPrevTotal.noalias() += uz_.value.transpose() * daz;

    dhPrev = dhPrevTotal;
}

void GRULayer::forwardInference(const Eigen::Ref<const Eigen::VectorXf>& x,
                                Eigen::VectorXf& h,
                                Eigen::VectorXf& z,
                                Eigen::VectorXf& r,
                                Eigen::VectorXf& n,
                                Eigen::VectorXf& scratch) const {
    z.noalias() = wz_.value * x;
    z.noalias() += uz_.value * h;
    z += bz_.value.col(0);
    z = (1.0f + (-z.array()).exp()).inverse().matrix();

    r.noalias() = wr_.value * x;
    r.noalias() += ur_.value * h;
    r += br_.value.col(0);
    r = (1.0f + (-r.array()).exp()).inverse().matrix();

    scratch = (r.array() * h.array()).matrix();
    n.noalias() = wn_.value * x;
    n.noalias() += un_.value * scratch;
    n += bn_.value.col(0);
    n = n.array().tanh();

    scratch = ((1.0f - z.array()) * n.array() + z.array() * h.array()).matrix();
    h.swap(scratch);
}

DenseLayer::DenseLayer(int inputSize, int outputSize, std::mt19937& rng) {
    initParam(w_, outputSize, inputSize, rng, xavierLimit(inputSize, outputSize));
    b_ = Param(outputSize, 1);
}

std::vector<Param*> DenseLayer::parameters() {
    return {&w_, &b_};
}

std::vector<const Param*> DenseLayer::parameters() const {
    return {&w_, &b_};
}

InferenceState::InferenceState(const ModelConfig& cfg) {
    resize(cfg);
}

void InferenceState::resize(const ModelConfig& cfg) {
    h1 = Eigen::VectorXf::Zero(cfg.hidden1);
    h2 = Eigen::VectorXf::Zero(cfg.hidden2);
    xNorm = Eigen::VectorXf::Zero(cfg.inputSize);
    z1 = Eigen::VectorXf::Zero(cfg.hidden1);
    r1 = Eigen::VectorXf::Zero(cfg.hidden1);
    n1 = Eigen::VectorXf::Zero(cfg.hidden1);
    scratch1 = Eigen::VectorXf::Zero(cfg.hidden1);
    z2 = Eigen::VectorXf::Zero(cfg.hidden2);
    r2 = Eigen::VectorXf::Zero(cfg.hidden2);
    n2 = Eigen::VectorXf::Zero(cfg.hidden2);
    scratch2 = Eigen::VectorXf::Zero(cfg.hidden2);
    densePre = Eigen::VectorXf::Zero(cfg.dense);
    denseAct = Eigen::VectorXf::Zero(cfg.dense);
}

void InferenceState::reset() {
    h1.setZero();
    h2.setZero();
    xNorm.setZero();
    z1.setZero();
    r1.setZero();
    n1.setZero();
    scratch1.setZero();
    z2.setZero();
    r2.setZero();
    n2.setZero();
    scratch2.setZero();
    densePre.setZero();
    denseAct.setZero();
}

NeuralNet::NeuralNet() : NeuralNet(ModelConfig{}, 1) {}

NeuralNet::NeuralNet(ModelConfig cfg, std::uint32_t seed) : cfg_(cfg) {
    if (cfg.inputSize <= 0 || cfg.hidden1 <= 0 || cfg.hidden2 <= 0 || cfg.dense <= 0) {
        throw std::runtime_error("invalid model dimensions");
    }
    std::mt19937 rng(seed);
    gru1_ = GRULayer(cfg.inputSize, cfg.hidden1, rng);
    gru2_ = GRULayer(cfg.hidden1, cfg.hidden2, rng);
    shaper_ = DenseLayer(cfg.hidden2, cfg.dense, rng);
    output_ = DenseLayer(cfg.dense, 1, rng);
    normMean_ = Eigen::VectorXf::Zero(cfg.inputSize);
    normStd_ = Eigen::VectorXf::Ones(cfg.inputSize);
}

std::vector<Param*> NeuralNet::parameters() {
    std::vector<Param*> out;
    appendParams(out, gru1_.parameters());
    appendParams(out, gru2_.parameters());
    appendParams(out, shaper_.parameters());
    appendParams(out, output_.parameters());
    return out;
}

std::vector<const Param*> NeuralNet::parameters() const {
    std::vector<const Param*> out;
    appendParams(out, gru1_.parameters());
    appendParams(out, gru2_.parameters());
    appendParams(out, shaper_.parameters());
    appendParams(out, output_.parameters());
    return out;
}

void NeuralNet::setNormalization(const Eigen::VectorXf& mean, const Eigen::VectorXf& stddev) {
    if (mean.size() != cfg_.inputSize || stddev.size() != cfg_.inputSize) {
        throw std::runtime_error("normalization dimensions do not match model input size");
    }
    normMean_ = mean;
    normStd_ = stddev.array().max(1.0e-6f);
}

Eigen::VectorXf NeuralNet::normalizeInputColumn(const Eigen::MatrixXf& input, int column) const {
    return ((input.col(column) - normMean_).array() / normStd_.array()).matrix();
}

InferenceState NeuralNet::makeInferenceState() const {
    return InferenceState(cfg_);
}

float NeuralNet::predictSample(const Eigen::Ref<const Eigen::VectorXf>& input,
                               InferenceState& state) const {
    if (input.size() != cfg_.inputSize) {
        throw std::runtime_error("predict sample input channel count does not match model");
    }
    if (state.xNorm.size() != cfg_.inputSize || state.h1.size() != cfg_.hidden1 ||
        state.h2.size() != cfg_.hidden2 || state.densePre.size() != cfg_.dense) {
        state.resize(cfg_);
    }

    state.xNorm = ((input - normMean_).array() / normStd_.array()).matrix();
    gru1_.forwardInference(state.xNorm, state.h1, state.z1, state.r1, state.n1, state.scratch1);
    gru2_.forwardInference(state.h1, state.h2, state.z2, state.r2, state.n2, state.scratch2);

    state.densePre.noalias() = shaper_.w_.value * state.h2;
    state.densePre += shaper_.b_.value.col(0);
    state.denseAct =
        (state.densePre.array() > 0.0f)
            .select(state.densePre.array(), state.densePre.array().exp() - 1.0f)
            .matrix();
    return (output_.w_.value * state.denseAct + output_.b_.value.col(0))(0);
}

Eigen::RowVectorXf NeuralNet::predict(const Eigen::MatrixXf& input) const {
    if (input.rows() != cfg_.inputSize) {
        throw std::runtime_error("predict input channel count does not match model");
    }
    Eigen::RowVectorXf prediction(input.cols());
    InferenceState state = makeInferenceState();
    for (int t = 0; t < input.cols(); ++t) {
        prediction(t) = predictSample(input.col(t), state);
    }
    return prediction;
}

TrainWindowResult NeuralNet::accumulateGradients(const Eigen::MatrixXf& input,
                                                 const Eigen::RowVectorXf& target,
                                                 int start, int length,
                                                 const Eigen::VectorXf& h1Initial,
                                                 const Eigen::VectorXf& h2Initial) {
    if (length <= 0 || start < 0 || start + length > input.cols() || target.size() < start + length) {
        throw std::runtime_error("invalid training window");
    }

    struct DenseCache {
        Eigen::VectorXf pre;
        Eigen::VectorXf activated;
    };

    std::vector<GRULayer::StepCache> c1(static_cast<std::size_t>(length));
    std::vector<GRULayer::StepCache> c2(static_cast<std::size_t>(length));
    std::vector<DenseCache> denseCache(static_cast<std::size_t>(length));
    std::vector<float> y(static_cast<std::size_t>(length), 0.0f);

    Eigen::VectorXf h1 = h1Initial;
    Eigen::VectorXf h2 = h2Initial;
    float loss = 0.0f;
    for (int local = 0; local < length; ++local) {
        const int t = start + local;
        h1 = gru1_.forwardStep(normalizeInputColumn(input, t), h1, c1[static_cast<std::size_t>(local)]);
        h2 = gru2_.forwardStep(h1, h2, c2[static_cast<std::size_t>(local)]);
        denseCache[static_cast<std::size_t>(local)].pre =
            shaper_.w_.value * h2 + shaper_.b_.value.col(0);
        denseCache[static_cast<std::size_t>(local)].activated =
            elu(denseCache[static_cast<std::size_t>(local)].pre);
        y[static_cast<std::size_t>(local)] =
            (output_.w_.value * denseCache[static_cast<std::size_t>(local)].activated +
             output_.b_.value.col(0))(0);
        const float diff = y[static_cast<std::size_t>(local)] - target(t);
        loss += diff * diff / static_cast<float>(length);
    }

    std::vector<Eigen::VectorXf> dh2(static_cast<std::size_t>(length),
                                     Eigen::VectorXf::Zero(cfg_.hidden2));
    for (int local = 0; local < length; ++local) {
        const int t = start + local;
        const float dy = 2.0f * (y[static_cast<std::size_t>(local)] - target(t)) /
                         static_cast<float>(length);
        const Eigen::VectorXf& activated = denseCache[static_cast<std::size_t>(local)].activated;
        output_.w_.grad.noalias() += Eigen::MatrixXf::Constant(1, 1, dy) * activated.transpose();
        output_.b_.grad(0, 0) += dy;
        Eigen::VectorXf dActivated = output_.w_.value.row(0).transpose() * dy;
        Eigen::VectorXf dPre =
            (dActivated.array() *
             eluDerivative(denseCache[static_cast<std::size_t>(local)].pre).array())
                .matrix();
        shaper_.w_.grad.noalias() += dPre * c2[static_cast<std::size_t>(local)].h.transpose();
        shaper_.b_.grad.col(0).noalias() += dPre;
        dh2[static_cast<std::size_t>(local)].noalias() += shaper_.w_.value.transpose() * dPre;
    }

    std::vector<Eigen::VectorXf> dh1(static_cast<std::size_t>(length),
                                     Eigen::VectorXf::Zero(cfg_.hidden1));
    Eigen::VectorXf dh2Next = Eigen::VectorXf::Zero(cfg_.hidden2);
    for (int local = length - 1; local >= 0; --local) {
        Eigen::VectorXf dx = Eigen::VectorXf::Zero(cfg_.hidden1);
        Eigen::VectorXf dhPrev = Eigen::VectorXf::Zero(cfg_.hidden2);
        gru2_.backwardStep(c2[static_cast<std::size_t>(local)],
                           dh2[static_cast<std::size_t>(local)] + dh2Next, dx, dhPrev);
        dh1[static_cast<std::size_t>(local)].noalias() += dx;
        dh2Next = dhPrev;
    }

    Eigen::VectorXf dh1Next = Eigen::VectorXf::Zero(cfg_.hidden1);
    for (int local = length - 1; local >= 0; --local) {
        Eigen::VectorXf dx = Eigen::VectorXf::Zero(cfg_.inputSize);
        Eigen::VectorXf dhPrev = Eigen::VectorXf::Zero(cfg_.hidden1);
        gru1_.backwardStep(c1[static_cast<std::size_t>(local)],
                           dh1[static_cast<std::size_t>(local)] + dh1Next, dx, dhPrev);
        dh1Next = dhPrev;
    }

    return {loss, h1, h2};
}

float NeuralNet::lossOnly(const Eigen::MatrixXf& input, const Eigen::RowVectorXf& target,
                          int start, int length) const {
    const Eigen::RowVectorXf prediction = predict(input.middleCols(start, length));
    float loss = 0.0f;
    for (int i = 0; i < length; ++i) {
        const float diff = prediction(i) - target(start + i);
        loss += diff * diff / static_cast<float>(length);
    }
    return loss;
}

void NeuralNet::zeroGradients() {
    for (Param* param : parameters()) {
        param->zeroGrad();
    }
}

void NeuralNet::scaleGradients(float scale) {
    for (Param* param : parameters()) {
        param->grad *= scale;
    }
}

float NeuralNet::gradientNorm() const {
    double sum = 0.0;
    for (const Param* param : parameters()) {
        sum += param->grad.squaredNorm();
    }
    return static_cast<float>(std::sqrt(sum));
}

void NeuralNet::clipGradients(float maxNorm) {
    if (maxNorm <= 0.0f) {
        return;
    }
    const float norm = gradientNorm();
    if (norm > maxNorm) {
        scaleGradients(maxNorm / (norm + 1.0e-6f));
    }
}

void NeuralNet::adamStep(float learningRate, std::int64_t step) {
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float eps = 1.0e-8f;
    const float bias1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bias2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (Param* param : parameters()) {
        param->m = beta1 * param->m + (1.0f - beta1) * param->grad;
        param->v = beta2 * param->v + (1.0f - beta2) * param->grad.array().square().matrix();
        const Eigen::MatrixXf mHat = param->m / bias1;
        const Eigen::MatrixXf vHat = param->v / bias2;
        param->value.array() -= learningRate * mHat.array() / (vHat.array().sqrt() + eps);
    }
}

} // namespace neural_amp
