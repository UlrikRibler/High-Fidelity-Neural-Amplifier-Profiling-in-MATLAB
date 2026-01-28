function [net, info] = TrainAmpModel(inputData, targetData, checkpointDir)
    %% TrainAmpModel.m (Gen 4)
    % ---------------------------------------------------------------------
    % GENERATION 4: 5-INPUT CONDITIONED TRAINING
    % ---------------------------------------------------------------------
    % Description:
    %   Trains a Stacked GRU to map [Audio, Gain, Bass, Mid, Treble] to Output.
    %   Optimized for 192kHz data (High VRAM load).
    %
    % Architecture:
    %   Input (5) -> GRU(128) -> GRU(64) -> Dense(32) -> ELU -> Output(1)
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------
    
    disp('Initializing Gen 4 Architecture (5-Input GRU)...');
    
    % GPU Setup
    hasPCT = ~isempty(ver('distcomp'));
    executionEnv = 'auto';
    if hasPCT, try, if gpuDeviceCount > 0, g=gpuDevice(1); reset(g); executionEnv='gpu'; end; catch, end; end
    
    % Checkpoints
    latestNet = [];
    if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end
    files = dir(fullfile(checkpointDir, 'net_checkpoint__*.mat'));
    if ~isempty(files)
        [~, idx] = max([files.datenum]);
        try, d=load(fullfile(checkpointDir, files(idx).name)); latestNet=d.net; disp('Resuming...'); catch, end
    end

    % Layers
    layers = [];
    if isempty(latestNet)
        layers = [ ...
            % Input: 5 Channels (Audio, Gain, Bass, Mid, Treble)
            sequenceInputLayer(5, 'Name', 'Input_FullStack', 'Normalization', 'zscore')
            
            % Gen 4: Wider First Layer (128 Units) to handle 192kHz complexity
            gruLayer(128, 'OutputMode', 'sequence', 'Name', 'GRU_Wideband')
            gruLayer(64,  'OutputMode', 'sequence', 'Name', 'GRU_Dynamics')
            
            fullyConnectedLayer(32, 'Name', 'Shaper')
            eluLayer('Name', 'NonLin')
            
            fullyConnectedLayer(1, 'Name', 'Output')
            regressionLayer('Name', 'Loss')
        ];
    end

    % Options (Tuned for 192kHz VRAM safety)
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', executionEnv, ...
        'MaxEpochs', 300, ...
        'MiniBatchSize', 64, ...           % Reduced to 64 for 192kHz (Prevents OOM)
        'SequenceLength', 'longest', ...
        'InitialLearnRate', 0.005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 50, ...
        'LearnRateDropFactor', 0.5, ...
        'Shuffle', 'every-epoch', ...
        'CheckpointPath', checkpointDir, ...
        'DispatchInBackground', true, ...
        'Plots', 'training-progress', ...
        'Verbose', true);
    
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