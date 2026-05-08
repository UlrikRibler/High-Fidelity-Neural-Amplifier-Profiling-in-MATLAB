#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace neural_amp {
namespace {

constexpr float kPi = 3.14159265358979323846f;

Biquad makeCookbook(float b0, float b1, float b2, float a0, float a1, float a2) {
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
}

float clampCutoff(float sampleRate, float cutoffHz) {
    const float nyquist = sampleRate * 0.5f;
    return std::clamp(cutoffHz, 1.0f, nyquist * 0.95f);
}

} // namespace

Biquad::Biquad(float b0, float b1, float b2, float a1, float a2)
    : b0_(b0), b1_(b1), b2_(b2), a1_(a1), a2_(a2) {}

Biquad Biquad::lowpass(float sampleRate, float cutoffHz, float q) {
    const float w0 = 2.0f * kPi * clampCutoff(sampleRate, cutoffHz) / sampleRate;
    const float cosW0 = std::cos(w0);
    const float alpha = std::sin(w0) / (2.0f * q);
    return makeCookbook((1.0f - cosW0) * 0.5f, 1.0f - cosW0, (1.0f - cosW0) * 0.5f,
                        1.0f + alpha, -2.0f * cosW0, 1.0f - alpha);
}

Biquad Biquad::highpass(float sampleRate, float cutoffHz, float q) {
    const float w0 = 2.0f * kPi * clampCutoff(sampleRate, cutoffHz) / sampleRate;
    const float cosW0 = std::cos(w0);
    const float alpha = std::sin(w0) / (2.0f * q);
    return makeCookbook((1.0f + cosW0) * 0.5f, -(1.0f + cosW0), (1.0f + cosW0) * 0.5f,
                        1.0f + alpha, -2.0f * cosW0, 1.0f - alpha);
}

Biquad Biquad::bandpass(float sampleRate, float centerHz, float q) {
    const float w0 = 2.0f * kPi * clampCutoff(sampleRate, centerHz) / sampleRate;
    const float alpha = std::sin(w0) / (2.0f * q);
    return makeCookbook(alpha, 0.0f, -alpha, 1.0f + alpha, -2.0f * std::cos(w0),
                        1.0f - alpha);
}

float Biquad::process(float x) {
    const float y = b0_ * x + z1_;
    z1_ = b1_ * x - a1_ * y + z2_;
    z2_ = b2_ * x - a2_ * y;
    return y;
}

void Biquad::reset() {
    z1_ = 0.0f;
    z2_ = 0.0f;
}

std::vector<float> logChirp(int samples, int sampleRate, float durationSeconds) {
    if (samples <= 0 || sampleRate <= 0 || durationSeconds <= 0.0f) {
        throw std::runtime_error("invalid chirp configuration");
    }

    constexpr float f0 = 20.0f;
    const float f1 = std::min(20000.0f, static_cast<float>(sampleRate) * 0.45f);
    const float sweepSeconds = std::max(durationSeconds / 3.0f, 0.001f);
    const float k = std::pow(f1 / f0, 1.0f / sweepSeconds);
    const float logK = std::log(k);

    std::vector<float> out(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const float t = std::min(static_cast<float>(i) / static_cast<float>(sampleRate), sweepSeconds);
        const float phase = 2.0f * kPi * f0 * ((std::pow(k, t) - 1.0f) / logK);
        out[static_cast<std::size_t>(i)] = std::sin(phase);
    }
    return out;
}

std::vector<float> pinkNoise(int samples, std::mt19937& rng) {
    if (samples <= 0) {
        throw std::runtime_error("invalid noise length");
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> out(static_cast<std::size_t>(samples));

    // Paul Kellet-style IIR approximation, enough for excitation data.
    float b0 = 0.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    float b3 = 0.0f;
    float b4 = 0.0f;
    float b5 = 0.0f;
    float b6 = 0.0f;
    for (int i = 0; i < samples; ++i) {
        const float white = normal(rng);
        b0 = 0.99886f * b0 + white * 0.0555179f;
        b1 = 0.99332f * b1 + white * 0.0750759f;
        b2 = 0.96900f * b2 + white * 0.1538520f;
        b3 = 0.86650f * b3 + white * 0.3104856f;
        b4 = 0.55000f * b4 + white * 0.5329522f;
        b5 = -0.7616f * b5 - white * 0.0168980f;
        const float pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362f;
        b6 = white * 0.115926f;
        out[static_cast<std::size_t>(i)] = pink * 0.11f;
    }
    normalizePeak(out);
    return out;
}

std::vector<float> randomKnob(int samples, float sampleRate, float speedHz, std::mt19937& rng) {
    if (samples <= 0 || sampleRate <= 0.0f || speedHz <= 0.0f) {
        throw std::runtime_error("invalid knob configuration");
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const float alpha = 1.0f - std::exp(-2.0f * kPi * speedHz / sampleRate);
    std::vector<float> out(static_cast<std::size_t>(samples));

    float state = 0.0f;
    float minValue = std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::lowest();
    for (int i = 0; i < samples; ++i) {
        state += alpha * (normal(rng) - state);
        out[static_cast<std::size_t>(i)] = state;
        minValue = std::min(minValue, state);
        maxValue = std::max(maxValue, state);
    }

    const float span = std::max(maxValue - minValue, 1.0e-6f);
    for (float& value : out) {
        value = (value - minValue) / span;
    }
    return out;
}

void normalizePeak(std::vector<float>& values, float epsilon) {
    float peak = 0.0f;
    for (float value : values) {
        peak = std::max(peak, std::abs(value));
    }
    const float denom = peak + epsilon;
    for (float& value : values) {
        value /= denom;
    }
}

bool allFinite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

} // namespace neural_amp
