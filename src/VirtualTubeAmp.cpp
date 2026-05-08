#include "VirtualTubeAmp.h"

#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neural_amp {

std::vector<float> processVirtualTubeAmp(const std::vector<float>& inputAudio,
                                         const std::vector<std::vector<float>>& bandControls,
                                         int sampleRate,
                                         float minBandHz,
                                         float maxBandHz) {
    const std::size_t n = inputAudio.size();
    if (n == 0 || bandControls.empty() || sampleRate <= 0) {
        throw std::runtime_error("invalid virtual amp inputs");
    }
    for (const std::vector<float>& band : bandControls) {
        if (band.size() != n) {
            throw std::runtime_error("band control length does not match input audio length");
        }
    }

    std::vector<float> stage1(n);
    const float dt = 1.0f / static_cast<float>(sampleRate);
    const float rc = 1.0f / (2.0f * 3.14159265358979323846f * 40.0f);
    const float hpAlpha = rc / (rc + dt);
    float hpY = 0.0f;
    float hpX = 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        constexpr float drive = 6.5f;
        const float clipped = std::tanh(inputAudio[i] * drive + 0.5f);
        hpY = hpAlpha * (hpY + clipped - hpX);
        hpX = clipped;
        stage1[i] = hpY;
    }

    const int bandCount = static_cast<int>(bandControls.size());
    const float nyquistSafe = static_cast<float>(sampleRate) * 0.45f;
    const float lowHz = std::clamp(minBandHz, 10.0f, nyquistSafe * 0.5f);
    const float highHz = std::clamp(maxBandHz, lowHz * 1.1f, nyquistSafe);
    const float logLow = std::log(lowHz);
    const float logHigh = std::log(highHz);
    std::vector<Biquad> filters;
    filters.reserve(static_cast<std::size_t>(bandCount));
    for (int band = 0; band < bandCount; ++band) {
        const float edge0 = std::exp(logLow + (logHigh - logLow) *
                                                  static_cast<float>(band) /
                                                  static_cast<float>(bandCount));
        const float edge1 = std::exp(logLow + (logHigh - logLow) *
                                                  static_cast<float>(band + 1) /
                                                  static_cast<float>(bandCount));
        const float center = std::sqrt(edge0 * edge1);
        const float q = std::clamp(center / std::max(edge1 - edge0, 1.0f), 0.35f, 12.0f);
        filters.push_back(Biquad::bandpass(static_cast<float>(sampleRate), center, q));
    }
    Biquad cab = Biquad::lowpass(static_cast<float>(sampleRate), 5000.0f);

    std::vector<float> output(n);
    const float bandNormalization = 1.0f / std::sqrt(static_cast<float>(bandCount));
    for (std::size_t i = 0; i < n; ++i) {
        float tone = 0.0f;
        for (int band = 0; band < bandCount; ++band) {
            const float filtered = filters[static_cast<std::size_t>(band)].process(stage1[i]);
            const float gain = 0.1f + bandControls[static_cast<std::size_t>(band)][i] * 3.9f;
            tone += filtered * gain;
        }
        tone *= bandNormalization;
        const float powered = tone / std::sqrt(1.0f + tone * tone);
        output[i] = cab.process(powered);
    }

    normalizePeak(output);
    return output;
}

} // namespace neural_amp
