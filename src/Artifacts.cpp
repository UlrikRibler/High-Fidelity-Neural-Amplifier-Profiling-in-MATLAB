#include "Artifacts.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace stock_signal {
namespace {

constexpr int kVersion = 1;
constexpr char kModelMagic[8] = {'N', 'A', 'M', 'O', 'D', 'E', 'L', '1'};
constexpr char kCheckpointMagic[8] = {'N', 'A', 'C', 'H', 'K', 'P', 'T', '1'};

void writeBytes(std::ostream& out, const void* data, std::size_t size) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error("failed to write artifact");
    }
}

void readBytes(std::istream& in, void* data, std::size_t size) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read artifact");
    }
}

template <typename T>
void writeScalar(std::ostream& out, const T& value) {
    writeBytes(out, &value, sizeof(T));
}

template <typename T>
T readScalar(std::istream& in) {
    T value{};
    readBytes(in, &value, sizeof(T));
    return value;
}

void writeMatrix(std::ostream& out, const Eigen::MatrixXf& matrix) {
    const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
    const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
    writeScalar(out, rows);
    writeScalar(out, cols);
    writeBytes(out, matrix.data(), sizeof(float) * static_cast<std::size_t>(rows) *
                                      static_cast<std::size_t>(cols));
}

Eigen::MatrixXf readMatrix(std::istream& in) {
    const auto rows = readScalar<std::int32_t>(in);
    const auto cols = readScalar<std::int32_t>(in);
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("invalid matrix dimensions in artifact");
    }
    Eigen::MatrixXf matrix(rows, cols);
    readBytes(in, matrix.data(), sizeof(float) * static_cast<std::size_t>(rows) *
                                      static_cast<std::size_t>(cols));
    return matrix;
}

void writeParam(std::ostream& out, const Param& param, bool optimizerState) {
    writeMatrix(out, param.value);
    if (optimizerState) {
        writeMatrix(out, param.m);
        writeMatrix(out, param.v);
    }
}

void readParam(std::istream& in, Param& param, bool optimizerState) {
    param.value = readMatrix(in);
    param.grad = Eigen::MatrixXf::Zero(param.value.rows(), param.value.cols());
    if (optimizerState) {
        param.m = readMatrix(in);
        param.v = readMatrix(in);
    } else {
        param.m = Eigen::MatrixXf::Zero(param.value.rows(), param.value.cols());
        param.v = Eigen::MatrixXf::Zero(param.value.rows(), param.value.cols());
    }
}

void writeNet(std::ostream& out, const NeuralNet& net, bool optimizerState) {
    const ModelConfig cfg = net.modelConfig();
    writeScalar(out, static_cast<std::int32_t>(cfg.inputSize));
    writeScalar(out, static_cast<std::int32_t>(cfg.hidden1));
    writeScalar(out, static_cast<std::int32_t>(cfg.hidden2));
    writeScalar(out, static_cast<std::int32_t>(cfg.dense));
    writeMatrix(out, net.normalizationMean());
    writeMatrix(out, net.normalizationStd());

    for (const Param* param : net.parameters()) {
        writeParam(out, *param, optimizerState);
    }
}

NeuralNet readNet(std::istream& in, bool optimizerState) {
    ModelConfig cfg;
    cfg.inputSize = readScalar<std::int32_t>(in);
    cfg.hidden1 = readScalar<std::int32_t>(in);
    cfg.hidden2 = readScalar<std::int32_t>(in);
    cfg.dense = readScalar<std::int32_t>(in);
    NeuralNet net(cfg, 1);
    const Eigen::MatrixXf mean = readMatrix(in);
    const Eigen::MatrixXf stddev = readMatrix(in);
    net.setNormalization(mean.col(0), stddev.col(0));

    for (Param* param : net.parameters()) {
        readParam(in, *param, optimizerState);
    }
    return net;
}

void checkMagic(std::istream& in, const char expected[8]) {
    char actual[8]{};
    readBytes(in, actual, sizeof(actual));
    for (int i = 0; i < 8; ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error("artifact magic header does not match");
        }
    }
}

} // namespace

void saveModel(const std::filesystem::path& path, const NeuralNet& net) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open model for writing: " + path.string());
    }
    writeBytes(out, kModelMagic, sizeof(kModelMagic));
    writeScalar(out, kVersion);
    writeNet(out, net, false);
}

NeuralNet loadModel(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open model: " + path.string());
    }
    checkMagic(in, kModelMagic);
    const int version = readScalar<int>(in);
    if (version != kVersion) {
        throw std::runtime_error("unsupported model version");
    }
    return readNet(in, false);
}

void saveCheckpoint(const std::filesystem::path& path, const NeuralNet& net, int epoch,
                    std::int64_t optimizerStep) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path.string());
    }
    writeBytes(out, kCheckpointMagic, sizeof(kCheckpointMagic));
    writeScalar(out, kVersion);
    writeScalar(out, static_cast<std::int32_t>(epoch));
    writeScalar(out, optimizerStep);
    writeNet(out, net, true);
}

CheckpointData loadCheckpoint(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open checkpoint: " + path.string());
    }
    checkMagic(in, kCheckpointMagic);
    const int version = readScalar<int>(in);
    if (version != kVersion) {
        throw std::runtime_error("unsupported checkpoint version");
    }
    CheckpointData data;
    data.epoch = readScalar<std::int32_t>(in);
    data.optimizerStep = readScalar<std::int64_t>(in);
    data.net = readNet(in, true);
    return data;
}

std::string inspectCheckpoint(const std::filesystem::path& path) {
    const CheckpointData data = loadCheckpoint(path);
    const ModelConfig cfg = data.net.modelConfig();
    std::ostringstream out;
    out << "Checkpoint: " << path.string() << '\n';
    out << "Epoch: " << data.epoch << '\n';
    out << "Optimizer step: " << data.optimizerStep << '\n';
    out << "Architecture: input " << cfg.inputSize << " -> GRU(" << cfg.hidden1
        << ") -> GRU(" << cfg.hidden2 << ") -> Dense(" << cfg.dense << ") -> output 1";
    return out.str();
}

} // namespace stock_signal
