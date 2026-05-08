#include "MarketSignalProcessor.h"

#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stock_signal {
namespace {

constexpr float kLowestAudibleBandHz = 20.0f;

} // namespace

std::vector<float> processMarketSignal(const std::vector<float>& inputSignal,
                                       const std::vector<std::vector<float>>& bandControls,
                                       int sampleRate,
                                       float minBandHz,
                                       float maxBandHz) {
    const std::size_t n = inputSignal.size();
    if (n == 0 || bandControls.empty() || sampleRate <= 0) {
        throw std::runtime_error("invalid market signal processor inputs");
    }
    for (const std::vector<float>& band : bandControls) {
        if (band.size() != n) {
            throw std::runtime_error("band control length does not match input signal length");
        }
    }

    const int bandCount = static_cast<int>(bandControls.size());
    const float nyquistSafe = static_cast<float>(sampleRate) * 0.45f;
    const float lowHz = std::max(0.0f, minBandHz);
    const float highHz = std::clamp(maxBandHz, 1.0f, nyquistSafe);
    if (highHz <= lowHz) {
        throw std::runtime_error("invalid market signal frequency range");
    }

    std::vector<Biquad> filters;
    filters.reserve(static_cast<std::size_t>(bandCount));

    const bool includeDcBand = lowHz <= 0.0f;
    const int logBandStart = includeDcBand ? 1 : 0;
    const int logBandCount = std::max(1, bandCount - logBandStart);
    const float logLow = std::log(std::max(kLowestAudibleBandHz, lowHz));
    const float logHigh = std::log(highHz);

    if (includeDcBand) {
        const float firstCutoff =
            bandCount == 1 ? highHz : std::min(highHz, std::max(1.0f, std::exp(logLow)));
        filters.push_back(Biquad::lowpass(static_cast<float>(sampleRate), firstCutoff));
    }

    for (int band = logBandStart; band < bandCount; ++band) {
        const float position0 = static_cast<float>(band - logBandStart) /
                                static_cast<float>(logBandCount);
        const float position1 = static_cast<float>(band - logBandStart + 1) /
                                static_cast<float>(logBandCount);
        const float edge0 = std::exp(logLow + (logHigh - logLow) * position0);
        const float edge1 = std::exp(logLow + (logHigh - logLow) * position1);
        const float center = std::sqrt(edge0 * edge1);
        const float q = std::clamp(center / std::max(edge1 - edge0, 1.0f), 0.35f, 16.0f);
        filters.push_back(Biquad::bandpass(static_cast<float>(sampleRate), center, q));
    }

    std::vector<float> output(n);
    const float bandNormalization = 1.0f / std::sqrt(static_cast<float>(bandCount));
    for (std::size_t i = 0; i < n; ++i) {
        float spectrum = 0.0f;
        for (int band = 0; band < bandCount; ++band) {
            const float filtered = filters[static_cast<std::size_t>(band)].process(inputSignal[i]);
            const float emphasis = 0.15f + bandControls[static_cast<std::size_t>(band)][i] * 2.85f;
            spectrum += filtered * emphasis;
        }
        spectrum *= bandNormalization;
        output[i] = spectrum / std::sqrt(1.0f + spectrum * spectrum);
    }

    normalizePeak(output);
    return output;
}

} // namespace stock_signal
