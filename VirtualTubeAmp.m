function output = VirtualTubeAmp(inputAudio, inputGain, fs)
    %% VirtualTubeAmp.m
    % ---------------------------------------------------------------------
    % THE TARGET: VIRTUAL ANALOG SIMULATION
    % ---------------------------------------------------------------------
    % Description:
    %   Acts as the "Ground Truth" hardware for the Neural Network to capture.
    %   It simulates a simplified 2-stage Tube Amplifier topology.
    %
    % Circuit Physics:
    %   1. Pre-Amp: 12AX7 Triode simulation using Asymmetric Tanh.
    %   2. Inter-stage: High-pass coupling capacitor (cuts mud).
    %   3. Power-Amp: Push-Pull simulation with soft-hard clipping.
    %   4. Cabinet: 4x12 Impulse Response approximation (Low Pass).
    %
    % Conditioning:
    %   Accepts 'inputGain' to simulate the physical rotation of a 
    %   potentiometer, altering the drive into Stage 1.
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------

    % Ensure inputs are column vectors
    inputAudio = inputAudio(:);
    inputGain = inputGain(:);
    
    %% STAGE 1: PRE-AMP (The "Color")
    % Driven by the Input Gain knob.
    % Tanh provides soft saturation similar to vacuum tubes.
    bias1 = 0.5; % Asymmetry factor (Even harmonics)
    stage1 = tanh(inputAudio .* inputGain + bias1);
    
    %% STAGE 2: COUPLING (The "Tightness")
    % High-Pass Filter @ 60Hz.
    % Removes DC offset and muddy bass frequencies before the power amp.
    [b_high, a_high] = butter(1, 60/(fs/2), 'high');
    stage1_filtered = filter(b_high, a_high, stage1);
    
    %% STAGE 3: POWER AMP (The "Drive")
    % Fixed gain stage simulating the EL34/6L6 power tubes.
    % Uses a specialized soft-knee clipper function.
    fixed_gain_stage2 = 4.0;
    stage2 = fixed_gain_stage2 * stage1_filtered;
    
    % Algebraic Sigmoid Clipper
    output_raw = stage2 ./ (1 + abs(stage2).^2).^0.5; 
    
    %% STAGE 4: CABINET (The "Tone")
    % Simulates the physical roll-off of a guitar speaker (Celestion V30).
    % Low Pass @ 4kHz.
    [b_cab, a_cab] = butter(2, 4000/(fs/2));
    output = filter(b_cab, a_cab, output_raw);
    
    %% NORMALIZATION
    % Ensure output is within -1.0 to 1.0 (Digital FS)
    if max(abs(output)) > 0
        output = output / max(abs(output));
    end
end