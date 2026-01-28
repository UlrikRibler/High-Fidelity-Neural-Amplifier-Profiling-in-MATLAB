function [net, info] = TrainAmpModel(inputData, targetData, checkpointDir)
    %% TrainAmpModel.m
    % ---------------------------------------------------------------------
    % THE BRAIN: STACKED GRU TRAINING
    % ---------------------------------------------------------------------
    % Description:
    %   Defines and trains the Deep Learning architecture.
    %   Uses a "Conditioned Stacked GRU" topology inspired by NeuralDSP.
    %
    % Architecture:
    %   [Input: Audio + Gain] 
    %        |
    %   [GRU Layer 1: 96 Units] --> Captures High Freq / Fast Transients
    %        |
    %   [GRU Layer 2: 48 Units] --> Captures Low Freq / Power Sag
    %        |
    %   [Dense + ELU] ----------> Waveshaping / Soft Clipping
    %        |
    %   [Output: Audio]
    %
    % Optimizations:
    %   - GPU Acceleration (Automatic Detection)
    %   - Multi-threaded Data Prefetching
    %   - Automatic Checkpoint Resuming
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------
    
    disp('Initializing Stacked GRU Architecture (High Performance)...');
    
    %% 1. HARDWARE ACCELERATION
    hasPCT = ~isempty(ver('distcomp'));
    executionEnv = 'auto';
    if hasPCT
        try
            if gpuDeviceCount > 0
                g = gpuDevice(1);
                disp(['Targeting GPU: ', g.Name]);
                reset(g); % Clear VRAM to prevent Out-Of-Memory errors
                executionEnv = 'gpu';
            end
        catch
            disp('GPU detection failed. Fallback to CPU.');
        end
    end
    
    %% 2. CHECKPOINT MANAGEMENT
    % Logic to resume training if the process crashed or was stopped.
    latestNet = [];
    if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end
    
    files = dir(fullfile(checkpointDir, 'net_checkpoint__*.mat'));
    if ~isempty(files)
        [~, idx] = max([files.datenum]);
        try
            d = load(fullfile(checkpointDir, files(idx).name));
            if isfield(d, 'net')
                latestNet = d.net; 
                disp(['RESUMING TRAINING from Checkpoint: ', files(idx).name]); 
            end
        catch
            disp('Warning: Checkpoint found but unreadable. Starting Fresh.');
        end
    end

    %% 3. NETWORK DEFINITION
    % We only define the layers if we are NOT resuming.
    layers = [];
    if isempty(latestNet)
        layers = [ ...
            % Input Layer: 2 Channels (Audio, Gain)
            % 'zscore' normalization ensures inputs are scaled 0-1 for the GRU
            sequenceInputLayer(2, 'Name', 'Input', 'Normalization', 'zscore')
            
            % Layer 1: High Complexity GRU
            % 96 Hidden Units allow complex harmonic mapping
            gruLayer(96, 'OutputMode', 'sequence', 'Name', 'GRU_Fast_Response')
            
            % Layer 2: Tapered GRU (The "Stack")
            % 48 Units handles the "slow" dynamics fed by the first layer
            gruLayer(48, 'OutputMode', 'sequence', 'Name', 'GRU_Slow_Dynamics')
            
            % Shaping Layer
            % Maps the internal GRU state (48 dims) back to Audio (1 dim)
            fullyConnectedLayer(32, 'Name', 'Harmonic_Shaper')
            eluLayer('Name', 'Tube_NonLinearity') % ELU is smoother than ReLU
            
            fullyConnectedLayer(1, 'Name', 'Output_Mixer')
            regressionLayer('Name', 'Audio_Loss')
        ];
    end

    %% 4. HYPERPARAMETERS
    % Aggressive optimization settings for NVIDIA 4070
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', executionEnv, ...
        'MaxEpochs', 300, ...
        'MiniBatchSize', 128, ...          % High batch size for GPU efficiency
        'SequenceLength', 'longest', ...   
        'InitialLearnRate', 0.005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 50, ...     % Decays LR every 50 epochs
        'LearnRateDropFactor', 0.5, ...
        'Shuffle', 'every-epoch', ...      % Essential for generalization
        'CheckpointPath', checkpointDir, ...
        'DispatchInBackground', true, ...  % Uses CPU threads to fetch data while GPU trains
        'WorkerLoad', 1, ...               % 100% GPU Utilization
        'Plots', 'training-progress', ...
        'Verbose', true);
    
    %% 5. EXECUTION
    try
        if ~isempty(latestNet)
            [net, info] = trainNetwork(inputData, targetData, latestNet, options);
        else
            [net, info] = trainNetwork(inputData, targetData, layers, options);
        end
    catch ME
        rethrow(ME);
    end
end