#pragma once

#include <random>
#include <vector>

namespace neural_amp {

class Biquad {
public:
    Biquad(float b0, float b1, float b2, float a1, float a2);

    static Biquad lowpass(float sampleRate, float cutoffHz, float q = 0.70710678f);
    static Biquad highpass(float sampleRate, float cutoffHz, float q = 0.70710678f);
    static Biquad bandpass(float sampleRate, float centerHz, float q);

    float process(float x);
    void reset();

private:
    float b0_ = 1.0f;
    float b1_ = 0.0f;
    float b2_ = 0.0f;
    float a1_ = 0.0f;
    float a2_ = 0.0f;
    float z1_ = 0.0f;
    float z2_ = 0.0f;
};

std::vector<float> logChirp(int samples, int sampleRate, float durationSeconds);
std::vector<float> pinkNoise(int samples, std::mt19937& rng);
std::vector<float> randomKnob(int samples, float sampleRate, float speedHz, std::mt19937& rng);
void normalizePeak(std::vector<float>& values, float epsilon = 1.0e-6f);
bool allFinite(const std::vector<float>& values);

} // namespace neural_amp
