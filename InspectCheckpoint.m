% InspectCheckpoint.m
% Loads the latest checkpoint and reports training status.

checkpointDir = 'checkpoints';
if ~exist(checkpointDir, 'dir')
    disp('No checkpoints folder found.');
    return;
end

files = dir(fullfile(checkpointDir, 'net_checkpoint__*.mat'));
if isempty(files)
    disp('No checkpoint files found.');
    return;
end

% Sort by time
[~, idx] = max([files.datenum]);
latestFile = fullfile(checkpointDir, files(idx).name);
disp(['Analyzing Checkpoint: ', files(idx).name]);

try
    data = load(latestFile);
    if isfield(data, 'net')
        % Extract info if available in the net object or saved struct
        % Note: 'net' object inspection in CLI is limited, but we can check layers
        disp(['Layers: ', num2str(length(data.net.Layers))]);
        disp(['Input Size: ', num2str(data.net.Layers(1).InputSize)]);
        
        % If 'info' struct was saved (custom), we'd read it here. 
        % Standard trainNetwork checkpoints only save 'net' and internal state.
        % We can't easily see "Current Loss" from the standard checkpoint file without resuming.
        disp('Checkpoint is valid and loadable.');
    else
        disp('Checkpoint file structure unrecognized.');
    end
catch ME
    disp(['Error loading checkpoint: ', ME.message]);
end
