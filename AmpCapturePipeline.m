%% AmpCapturePipeline.m
% -------------------------------------------------------------------------
% NEURAL AMP CAPTURE: GENERATION 4 (GOD TIER)
% -------------------------------------------------------------------------
% Description:
%   The Master Orchestrator for the NeuralMat project.
%   Manages the 192kHz High-Fidelity pipeline with Full Control Conditioning.
%
% Workflow:
%   1. CONFIGURATION  : 192kHz, 5 Inputs, 128-Unit GRU.
%   2. DATA GENERATION: Synthesize 3 minutes of 192kHz audio + 4 Random Knobs.
%   3. TRAINING       : Train the 5-Input Network on GPU.
%   4. LOGGING        : "Black Box" Session recording.
%
% Architecture:
%   - Stacked GRU (128 -> 64 units)
%   - 5-Channel Input (Audio, Gain, Bass, Mid, Treble)
%
% Requirements:
%   - NVIDIA GPU (8GB+ VRAM)
%
% Author: NeuralMat Team
% License: MIT
% -------------------------------------------------------------------------

clc; clear; close all;

% =========================================================================
% %% 1. CONFIGURATION
% =========================================================================
cfg.fs = 192000;             % 192kHz (Mastering Grade - Anti-Aliasing)
cfg.duration = 180;          % 3 Minutes
cfg.epochs = 300;
cfg.batchSize = 64;          % Reduced to 64 to prevent VRAM overflow at 192k
cfg.architecture = 'Gen 4: 5-Input Stacked GRU (128/64)';
cfg.gpu = true;

% Session Setup
sessionTimestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
sessionID = ['Session_', sessionTimestamp, '_GEN4'];
baseDir = fullfile('experiments', sessionID);

dirs.checkpoints = fullfile(baseDir, 'checkpoints');
dirs.data = fullfile(baseDir, 'data');
dirs.plots = fullfile(baseDir, 'plots');
dirs.model = fullfile(baseDir);

mkdir(dirs.checkpoints); mkdir(dirs.data); mkdir(dirs.plots);

logFile = fullfile(baseDir, 'training_log.txt');
diary(logFile);

disp('================================================');
disp(['   NEURAL AMP CAPTURE: GENERATION 4 (GOD TIER)']);
disp(['   Sample Rate: 192kHz | Inputs: 5 (Full Stack)']);
disp('================================================');

cfgFile = fullfile(baseDir, 'config.mat');
save(cfgFile, 'cfg');

try
    %% 1. Generate Datasets
    disp('[Step 1] Generating 192kHz Multi-Knob Dataset...');
    [trainInput, trainTarget] = DataGenerator(cfg.fs, cfg.duration);
    
    disp('Saving Massive Data Artifacts (1GB+)...');
    dataFile = fullfile(dirs.data, 'dataset.mat');
    save(dataFile, 'trainInput', 'trainTarget', '-v7.3'); % v7.3 required for >2GB

    %% 2. Train Neural Network
    disp('[Step 2] Training Gen 4 Network...');
    [net, trainInfo] = TrainAmpModel(trainInput, trainTarget, dirs.checkpoints);

    modelFile = fullfile(dirs.model, 'final_model.mat');
    save(modelFile, 'net', 'trainInfo');

    %% 3. Validation
    disp('[Step 3] Validating Gen 4 Model...');
    idx = randi(length(trainInput));
    ModelValidator(net, trainInput{idx}, trainTarget{idx}, cfg.fs, dirs.plots);

    disp('GEN 4 CAPTURE COMPLETE.');

catch ME
    disp('CRITICAL FAILURE');
    disp(ME.message);
    disp(ME.stack(1));
end
diary off;