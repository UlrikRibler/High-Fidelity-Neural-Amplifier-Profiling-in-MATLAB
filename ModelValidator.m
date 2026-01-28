function ModelValidator(net, inputFeatures, targetSignal, fs, outputDir)
    %% ModelValidator.m (Gen 4)
    % ---------------------------------------------------------------------
    % THE JUDGE: 192kHz VISUALIZATION
    % ---------------------------------------------------------------------
    % Description:
    %   Visualizes the waveform match and the state of all 4 Knobs.
    %   Validates that the AI follows the Gain/EQ changes correctly.
    %
    % Author: NeuralMat Team
    % ---------------------------------------------------------------------

    disp('Running Gen 4 Validation...');
    
    predictedSignal = predict(net, inputFeatures);
    errorSignal = targetSignal - predictedSignal;
    esr = sum(errorSignal.^2) / sum(targetSignal.^2);
    
    disp(sprintf('Validation ESR: %.5f (Accuracy: %.3f%%)', esr, (1-esr)*100));
    
    f = figure('Name', 'Gen 4 Capture Results', 'NumberTitle', 'off', 'Visible', 'off');
    
    % Audio
    subplot(3,1,1);
    range = 10000:12000; % Zoom
    plot(range, targetSignal(range), 'b', 'LineWidth', 1.5); hold on;
    plot(range, predictedSignal(range), 'r--', 'LineWidth', 1.0);
    title(['192kHz Waveform Match (ESR: ', num2str(esr, '%.5f'), ')']);
    grid on;
    
    % Error
    subplot(3,1,2);
    plot(range, errorSignal(range), 'k');
    title('Residual Error (High Res)');
    grid on;
    
    % Knobs
    subplot(3,1,3);
    % Plot all 4 conditioning signals
    hold on;
    plot(inputFeatures(2, range), 'r', 'DisplayName', 'Gain');
    plot(inputFeatures(3, range), 'g', 'DisplayName', 'Bass');
    plot(inputFeatures(4, range), 'b', 'DisplayName', 'Mid');
    plot(inputFeatures(5, range), 'c', 'DisplayName', 'Treble');
    legend('show');
    title('Multi-Knob Conditioning State');
    grid on;
    
    saveas(f, fullfile(outputDir, 'gen4_result.png'));
    close(f);
end