function [inputCell, targetCell] = DataGenerator(fs, durationSeconds)
    %% DataGenerator.m (Gen 4)
    % ---------------------------------------------------------------------
    % GENERATION 4: 192kHz MULTI-DIMENSIONAL DATASET
    % ---------------------------------------------------------------------
    % Description:
    %   Generates Ultra-High-Res audio and RANDOM WALKS for 4 Knobs.
    %   The network must learn to separate the effect of Bass from Gain.
    %
    % Output:
    %   inputCell:  {5 x N} Matrix (Audio, Gain, Bass, Mid, Treble)
    %   targetCell: {1 x N} Matrix (Output)
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------

    disp(['Generating ', num2str(durationSeconds), 's of 192kHz Mastering-Grade Audio...']);
    
    dt = 1/fs;
    N = round(fs * durationSeconds);
    t = (0:N-1)'/fs;
    
    %% 1. AUDIO (The "Source")
    % Mixed Signal: Chirps + Pink Noise + Impulses
    s1 = chirp(t, 20, durationSeconds/3, 20000, 'logarithmic');
    
    % Pink Noise (FFT)
    white = randn(N, 1);
    X = fft(white);
    numUniquePts = ceil((N+1)/2);
    n = 1:numUniquePts; n(1)=1;
    X(1:numUniquePts) = X(1:numUniquePts) .* (1./sqrt(n'));
    if rem(N, 2), X(numUniquePts+1:N) = conj(X(N:-1:numUniquePts+1));
    else, X(numUniquePts+1:N) = conj(X(numUniquePts-1:-1:2)); end
    pink = real(ifft(X));
    pink = pink / max(abs(pink));
    
    rawAudio = 0.5*s1 + 0.5*pink;
    rawAudio = rawAudio / max(abs(rawAudio));
    
    %% 2. KNOB TRAJECTORIES (The "Hand")
    % We need the knobs to move INDEPENDENTLY so the AI can learn what
    % each one does. If they all move together, it can't distinguish them.
    
    % Generate 4 Slow Random Walks (Low Pass Filtered Random Noise)
    disp('Synthesizing Knob Movements...');
    
    function k = makeKnob(len, speed)
        rw = randn(len, 1);
        [b, a] = butter(1, speed/(fs/2)); % Very slow filter
        k = filter(b, a, rw);
        % Normalize to 0-1
        k = (k - min(k)) / (max(k) - min(k));
    end

    k_gain   = makeKnob(N, 0.2); % 0.2Hz
    k_bass   = makeKnob(N, 0.3);
    k_mid    = makeKnob(N, 0.4);
    k_treble = makeKnob(N, 0.5);
    
    %% 3. GROUND TRUTH
    disp('Running Virtual 4-Knob Amp (High Load)...');
    targetAudio = VirtualTubeAmp(rawAudio, k_gain, k_bass, k_mid, k_treble, fs);
    
    %% 4. CHUNK & BATCH
    % 192kHz = 0.5s is 96,000 samples.
    chunkLen = round(0.5 * fs);
    hopLen   = round(0.25 * fs);
    numChunks = floor((N - chunkLen) / hopLen) + 1;
    
    disp(['Slicing into ', num2str(numChunks), ' High-Res segments...']);
    
    inputCell = cell(numChunks, 1);
    targetCell = cell(numChunks, 1);
    
    % Input Feature Matrix: 5 Rows x Time
    fullInput = [rawAudio'; k_gain'; k_bass'; k_mid'; k_treble'];
    fullTarget = targetAudio';
    
    for i = 1:numChunks
        idx = (i-1)*hopLen + 1;
        range = idx : idx + chunkLen - 1;
        inputCell{i}  = fullInput(:, range);
        targetCell{i} = fullTarget(:, range);
    end
    
    disp('Gen 4 Dataset Ready.');
end