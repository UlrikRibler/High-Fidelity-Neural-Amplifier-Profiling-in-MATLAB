function [inputCell, targetCell] = DataGenerator(fs, durationSeconds)
    %% DataGenerator.m
    % ---------------------------------------------------------------------
    % GENERATION 3: GPU-SATURATING PROFILING SIGNAL GENERATOR
    % ---------------------------------------------------------------------
    % Description:
    %   Creates the "Training Syllabus" for the Neural Network.
    %   Instead of random noise, this generates a structured, scientific 
    %   excitation signal designed to fully map the state-space of a 
    %   non-linear audio device.
    %
    % Key Features:
    %   1. Logarithmic Sine Sweeps: Captures Frequency Response.
    %   2. Pink Noise (1/f): Captures Transient Response (Chaos).
    %   3. Guitar Envelope Simulation: Mimics pick attack dynamics.
    %   4. CONDITIONING: Sweeps a "Virtual Gain Knob" to teach the model
    %      how the amp behaves at different drive levels.
    %
    % Output:
    %   inputCell:  Cell Array of {2 x SegmentLength} (Audio + Gain)
    %   targetCell: Cell Array of {1 x SegmentLength} (Amp Output)
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------

    disp(['Generating ', num2str(durationSeconds), ' seconds of High-Fidelity Audio...']);
    
    % Constants
    dt = 1/fs;
    N = round(fs * durationSeconds);
    t = (0:N-1)'/fs;
    
    %% 1. EXCITATION SIGNAL SYNTHESIS
    % We blend multiple signal types to ensure "spectral richness".
    
    % A. Logarithmic Chirps (The ESR Killer)
    % Sweeps 20Hz -> 20kHz exponentially. Ideally suited for audio systems.
    s1 = chirp(t, 20, durationSeconds/3, 20000, 'logarithmic');
    s2 = chirp(t, 20, durationSeconds/10, 20000, 'logarithmic'); % Fast repeats
    
    % B. Pink Noise Generation (The "Feel" Killer)
    % We use the FFT method to generate 1/f noise without external toolboxes.
    % Pink noise has equal energy per octave, similar to guitar signals.
    white = randn(N, 1);
    X = fft(white);
    numUniquePts = ceil((N+1)/2);
    n = 1:numUniquePts; n(1)=1;
    X(1:numUniquePts) = X(1:numUniquePts) .* (1./sqrt(n')); % 1/sqrt(f) filter
    
    % Enforce Hermitian Symmetry for Real IFFT
    if rem(N, 2)
        X(numUniquePts+1:N) = conj(X(N:-1:numUniquePts+1));
    else
        X(numUniquePts+1:N) = conj(X(numUniquePts-1:-1:2));
    end
    pink = real(ifft(X));
    pink = pink / max(abs(pink));
    
    % C. Simulated Guitar Strums
    % Filtered noise modulated by a 4Hz envelope to mimic picking.
    strum_env = abs(sin(2*pi*4*t)); 
    [b_git, a_git] = butter(2, [80 5000]/(fs/2)); % Guitar Cabinet Range
    guitar_sim = filter(b_git, a_git, white) .* strum_env;
    
    % Mix Components
    rawAudio = 0.4*s1 + 0.3*s2 + 0.2*pink + 0.4*guitar_sim;
    rawAudio = rawAudio / max(abs(rawAudio));
    
    %% 2. CONDITIONING (Virtual Knob)
    % We generate a control signal that sweeps the gain from Clean -> Metal.
    % This teaches the network the "concept" of a Gain Knob.
    % Range: 1.0 (Clean) to 11.0 (Distorted)
    knob_signal = 1 + 10 * (0.5 * (sin(2*pi*0.1*t) + 1)); % 0.1Hz Slow Sweep
    
    %% 3. GROUND TRUTH GENERATION
    % Pass the signals through the Target Device (Virtual or Real).
    disp('Processing through Virtual Amp (Physics Simulation)...');
    targetAudio = VirtualTubeAmp(rawAudio, knob_signal, fs);
    
    %% 4. SEGMENTATION & BATCHING
    % To saturate the GPU, we slice the long audio into thousands of
    % overlapping small chunks.
    
    chunkLen = round(0.5 * fs); % 0.5 Second Segments
    hopLen = round(0.25 * fs);  % 50% Overlap
    
    numChunks = floor((N - chunkLen) / hopLen) + 1;
    
    disp(['Slicing into ', num2str(numChunks), ' parallel training segments...']);
    
    inputCell = cell(numChunks, 1);
    targetCell = cell(numChunks, 1);
    
    % Combine Audio and Gain into Input Matrix
    fullInput = [rawAudio'; knob_signal']; % 2 x N (Features x Time)
    fullTarget = targetAudio';             % 1 x N
    
    for i = 1:numChunks
        startIdx = (i-1)*hopLen + 1;
        endIdx = startIdx + chunkLen - 1;
        
        inputCell{i} = fullInput(:, startIdx:endIdx);
        targetCell{i} = fullTarget(:, startIdx:endIdx);
    end
    
    disp('Dataset Generation Complete.');
end