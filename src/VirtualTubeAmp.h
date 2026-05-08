#pragma once

#include <vector>

namespace neural_amp {

std::vector<float> processVirtualTubeAmp(const std::vector<float>& inputAudio,
                                         const std::vector<std::vector<float>>& bandControls,
                                         int sampleRate,
                                         float minBandHz = 40.0f,
                                         float maxBandHz = 20000.0f);

} // namespace neural_amp
