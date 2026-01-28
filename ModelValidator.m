function ModelValidator(net, inputFeatures, targetSignal, fs, outputDir)
    %% ModelValidator.m
    % ---------------------------------------------------------------------
    % THE JUDGE: PERFORMANCE VISUALIZATION & METRICS
    % ---------------------------------------------------------------------
    % Description:
    %   Quantifies the accuracy of the trained neural network.
    %   Calculates the industry-standard ESR (Error-to-Signal Ratio) and
    %   generates high-resolution plots of the waveform match.
    %
    % Metrics:
    %   - ESR: < 0.01 is considered "Indistinguishable".
    %
    % Inputs:
    %   inputFeatures: 2 x N Matrix (Audio, Gain)
    %   targetSignal : 1 x N Matrix (Ground Truth)
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------
    
    disp('Running Validation...');
    
    %% 1. INFERENCE
    % Run the model in "Prediction Mode".
    predictedSignal = predict(net, inputFeatures);
    
    %% 2. METRIC CALCULATION
    % ESR = Sum(Error^2) / Sum(Target^2)
    errorSignal = targetSignal - predictedSignal;
    esr = sum(errorSignal.^2) / sum(targetSignal.^2);
    
    accuracy = (1-esr)*100;
    
    % Log score to console
    resultStr = sprintf('Validation ESR: %.4f (Accuracy: %.2f%%)', esr, accuracy);
    disp(resultStr);
    
    %% 3. VISUALIZATION
    % Create a rigorous comparison plot (invisible window for background saving)
    f = figure('Name', 'Conditioned Neural Amp Capture', 'NumberTitle', 'off', 'Visible', 'off');
    
    % Extract Audio Component (Row 1) for plotting x-axis
    inputAudio = inputFeatures(1, :);
    
    % PLOT A: Time Domain Zoom
    subplot(3,1,1);
    center = round(length(inputAudio)/2);
    range = center : center + 1000; % Zoom in on 1000 samples (~20ms)
    
    plot(range/fs, targetSignal(range), 'b', 'LineWidth', 1.5); hold on;
    plot(range/fs, predictedSignal(range), 'r--', 'LineWidth', 1.0);
    legend('Target (Truth)', 'Prediction (Neural)');
    title(['Stacked GRU Match (ESR: ', num2str(esr, '%.4f'), ')']);
    ylabel('Amplitude');
    grid on;
    
    % PLOT B: Residual Error
    subplot(3,1,2);
    plot(range/fs, errorSignal(range), 'k');
    title('Residual Error (Difference)');
    ylabel('Error Amplitude');
    grid on;
    
    % PLOT C: Conditioning Context
    subplot(3,1,3);
    gainCurve = inputFeatures(2, range);
    plot(range/fs, gainCurve, 'm', 'LineWidth', 2);
    title('Conditioning Input (Virtual Gain Knob)');
    ylabel('Gain Value');
    xlabel('Time (s)');
    grid on;
    
    %% 4. ARTIFACT SAVING
    saveFilename = fullfile(outputDir, 'validation_result.png');
    saveas(f, saveFilename);
    disp(['Validation plot saved to: ', saveFilename]);
    close(f);
end