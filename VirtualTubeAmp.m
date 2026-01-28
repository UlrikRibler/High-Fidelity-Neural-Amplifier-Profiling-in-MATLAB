function output = VirtualTubeAmp(inputAudio, gainKnob, bassKnob, midKnob, trebleKnob, fs)
    %% VirtualTubeAmp.m (Gen 4)
    % ---------------------------------------------------------------------
    % THE TARGET: 4-KNOB HIGH-FIDELITY TUBE SIMULATION
    % ---------------------------------------------------------------------
    % Description:
    %   A fully conditioned virtual amp with Gain + 3-Band Tone Stack.
    %   Running at 192kHz for anti-aliasing.
    %
    % Controls (0.0 - 1.0 Normalized):
    %   - Gain  : Pre-amp Drive
    %   - Bass  : Low Shelf Gain (+/- 12dB)
    %   - Mid   : Peaking Filter Gain (+/- 12dB @ 500Hz)
    %   - Treble: High Shelf Gain (+/- 12dB)
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------

    % Ensure inputs are column vectors
    inputAudio = inputAudio(:);
    gainKnob = gainKnob(:);
    bassKnob = bassKnob(:);
    midKnob = midKnob(:);
    trebleKnob = trebleKnob(:);
    
    %% 1. PRE-AMP (Distortion)
    % Map Gain Knob (0-1) to practical Drive (1.0 - 12.0)
    drive = 1.0 + (gainKnob * 11.0);
    
    % Asymmetric Tanh Clipping
    bias = 0.5; 
    stage1 = tanh(inputAudio .* drive + bias);
    
    % DC Block
    [b_hp, a_hp] = butter(1, 40/(fs/2), 'high');
    stage1 = filter(b_hp, a_hp, stage1);

    %% 2. TONE STACK (3-Band EQ)
    % We process the signal through 3 filters (Bass, Mid, Treble).
    % Since the knobs change over time, we cannot use static 'filter'.
    % We must approximate the tone stack by applying the filters to the 
    % whole signal but modulating the *Mix* or using variable gain approximation.
    
    % For computational efficiency in this "Ground Truth" generator,
    % we will use a simplified "Parallel Band" approach which allows
    % instantaneous mixing based on knob positions.
    
    % Filter Design
    % Low Band (< 200Hz)
    [b_low, a_low] = butter(2, 200/(fs/2), 'low');
    sig_low = filter(b_low, a_low, stage1);
    
    % Mid Band (200Hz - 2kHz)
    [b_mid, a_mid] = butter(2, [200 2000]/(fs/2), 'bandpass');
    sig_mid = filter(b_mid, a_mid, stage1);
    
    % High Band (> 2kHz)
    [b_high, a_high] = butter(2, 2000/(fs/2), 'high');
    sig_high = filter(b_high, a_high, stage1);
    
    % Apply Knob Gains (Mapping 0-1 to 0.1x - 4.0x gain)
    % Bass
    g_bass = 0.1 + (bassKnob * 3.9);
    % Mid (Scooped by default in amps, so 0.5 is neutral)
    g_mid  = 0.1 + (midKnob * 3.9);
    % Treble
    g_treb = 0.1 + (trebleKnob * 3.9);
    
    % Sum Bands
    stage2 = (sig_low .* g_bass) + (sig_mid .* g_mid) + (sig_high .* g_treb);
    
    %% 3. POWER AMP & CAB
    % Soft-Hard Clipper
    stage3 = stage2 ./ (1 + abs(stage2).^2).^0.5; 
    
    % Cabinet Impulse Response (simulated via filter)
    % 192kHz requires adjusting the cutoff normalization
    [b_cab, a_cab] = butter(2, 5000/(fs/2));
    output = filter(b_cab, a_cab, stage3);
    
    % Normalize
    output = output / (max(abs(output)) + 1e-6);
end