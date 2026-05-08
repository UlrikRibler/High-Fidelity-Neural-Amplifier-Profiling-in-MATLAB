#pragma once

#include <vector>

namespace stock_signal {

std::vector<float> processMarketSignal(const std::vector<float>& inputSignal,
                                       const std::vector<std::vector<float>>& bandControls,
                                       int sampleRate,
                                       float minBandHz = 0.0f,
                                       float maxBandHz = 20000.0f);

} // namespace stock_signal
