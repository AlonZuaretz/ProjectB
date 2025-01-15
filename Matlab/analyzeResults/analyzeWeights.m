clear
set(0, 'DefaultFigureWindowStyle', 'docked');
% load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV7\globalParams.mat")
% load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV7\NN_results\stage1_run_20241224_052213\test_results.mat")

load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\NN_results\stage1_run_20241210_141656\test_results.mat")

%%
% Filter:
% term1 = zeros(size(pythonParams,2),1); term2 = zeros(size(pythonParams,2),1); term3 = zeros(size(pythonParams,2),1); term4 = zeros(size(pythonParams,2),1);
% for i = 1:length(pythonParams)
%     term1(i) = pythonParams(i).SIR >= -30;
% end
% for i = 1:length(pythonParams)
%     term2(i) = pythonParams(i).SNR <= inf;
% end
% for i = 1:length(pythonParams)
%     term3(i) = abs(double(pythonParams(i).inputAngle(1)) - double(pythonParams(i).interferenceAngle(1))) >= 40;
%     term4(i) = abs(double(pythonParams(i).inputAngle(1)) - double(pythonParams(i).interferenceAngle(1))) <= 80;
% end
% terms = term1 .* term2 .* term3 .* term4;
% relIdxs = find(terms);
% pythonParams = pythonParams(relIdxs);
% label_YR = label_YR(relIdxs,:,:);
% label_XR = label_XR(relIdxs,:,:);
% input_XRd = input_XRd(relIdxs,:,:);
% output_YR = output_YR(relIdxs,:,:);
% output_XR = output_XR(relIdxs,:,:);
% Indexes = Indexes(relIdxs);

% sort by seed:
[B, I] = sort([pythonParams.seed]);
pythonParams = pythonParams(I);
label_YR = label_YR(I,:,:);
label_XR = label_XR(I,:,:);
input_XRd = input_XRd(I,:,:);
output_YR = output_YR(I,:,:);
output_XR = output_XR(I,:,:);
Indexes = Indexes(I);

%% Define some constants:
c = physconst('LightSpeed'); %  light speed in [ m/s ]
M = 4;
npc = 2;
carrierFreq = 28e9;
lambda = c/carrierFreq;
d = lambda/2;
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
thetaScan = -60:0.5:60;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for i = 1:length(thetaScan)
    theta = [thetaScan(i) ; 0];
    steeringVecMat(i, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end
%% Estimate Input Angle:
s(1:length(pythonParams)) = struct('index', [],'inputGain', [], 'interferenceGain', [], 'angleError', [], 'angleDiff', [], 'inputAngle', ...
        [], 'interferenceAngle', [], 'SIR', [], 'SNR', [], 'N', []);

for i = 1:length(pythonParams)
    pyParams = pythonParams(i);
    inputAngle = double(pyParams.inputAngle(1));
    interferenceAngle = double(pyParams.interferenceAngle(1));
    R = squeeze(output_XR(i,:,:));

    % Estimate angle of input:
    P_music = musicSpectEst(R, steeringVecMat); 
    [~, locs] = findpeaks(log10(P_music), thetaScan, 'SortStr', 'descend', 'MinPeakProminence', 1);
    if length(locs) <2
        s(i) = struct('index', NaN,'inputGain', NaN, 'interferenceGain', NaN, 'angleError', NaN, 'angleDiff', NaN, 'inputAngle', ...
        NaN, 'interferenceAngle', NaN, 'SIR', NaN, 'SNR', NaN, 'N', NaN);
        continue;
    end
    estAngle = locs(2);

    % Calculate weights:
    [w, R_pc] = pc_beamformer(R, npc, M, estAngle);

    % Calculate Weights pattern at inputAngle and interferenceAngle:
    wp = weightsPattern(w, steeringVecMat);
    inputGain = wp(thetaScan==inputAngle);
    interferenceGain = wp(thetaScan==interferenceAngle);
    angleError = abs(estAngle - double(pyParams.inputAngle(1)));
    angleDiff= abs(double(pyParams.inputAngle(1)) - double(pyParams.interferenceAngle(1)));

    s(i) = struct('index', i,'inputGain', inputGain, 'interferenceGain', interferenceGain, 'angleError', angleError, 'angleDiff', angleDiff, 'inputAngle', ...
        inputAngle, 'interferenceAngle', interferenceAngle, 'SIR', double(pyParams.SIR), 'SNR', double(pyParams.SNR), 'N', double(pyParams.N));
end

validIdxs = find(~isnan([s.index]));
s = s(validIdxs);
[~, I] = sort([s.interferenceGain], 'descend');
s = s(I);

%%
x = [s(:).SIR].';
y1 = [s(:).interferenceGain].';
y2 = [s(:).angleError].';

% Find unique angleDiff values and compute their average inputGain
[unique_x, ~, idx] = unique(x);       % Unique angleDiff values
avg_y1 = accumarray(idx, y1, [], @mean);
avg_y2 = accumarray(idx, y2, [], @mean);

% Define interpolation grid for angleDiff
x_grid = linspace(min(x), max(x), 100);

% Interpolate averaged inputGain onto a regular grid
y1_interp = interp1(unique_x, avg_y1, x_grid, 'linear');
y2_interp = interp1(unique_x, avg_y2, x_grid, 'linear');

% Plot the averaged and interpolated data
figure;
plot(x_grid, 10*log10(y1_interp), 'LineWidth', 1.5); hold on;
yyaxis right
plot(x_grid, y2_interp, 'LineWidth', 1.5);

xlabel('SIR [dB]', 'FontSize', 12);
grid on;

legend('interferenge Gain', 'Angle error')
%%

% Convert to dB scale
inputGaindB = 10*log10([s.inputGain]);
interferenceGaindB = 10*log10([s.interferenceGain]);

% Define thresholds
inputGainThreshold = 4; % Gain > 1
interferenceGainThreshold = -15; % Gain < -15 dB

% Classify the weights
isGood = (inputGaindB > inputGainThreshold) & (interferenceGaindB < interferenceGainThreshold);
isBad = (inputGaindB < 4) & (interferenceGaindB > 0);

% Calculate the difference between input and interference gains
gainDifference = inputGaindB - interferenceGaindB;

% Plot histograms
figure;

% Input Gain Histogram
subplot(3,1,1);
histogram(inputGaindB, 'Normalization', 'probability');
hold on;
xline(inputGainThreshold, 'r--', 'LineWidth', 1.5);
hold off;
title('Input Gain Distribution');
xlabel('Input Gain (dB)');
ylabel('Probability');
legend('Distribution', 'Threshold');
grid on;

% Interference Gain Histogram
subplot(3,1,2);
histogram(interferenceGaindB, 'Normalization', 'probability');
hold on;
xline(interferenceGainThreshold, 'r--', 'LineWidth', 1.5);
hold off;
title('Interference Gain Distribution');
xlabel('Interference Gain (dB)');
ylabel('Probability');
legend('Distribution', 'Threshold');
grid on;

% Gain Difference Histogram
subplot(3,1,3);
histogram([s.angleError], 'Normalization', 'probability');
title('Gain Difference Distribution');
xlabel('Gain Difference (Input - Interference) (dB)');
ylabel('Probability');
grid on;

% Summary Statistics
goodCount = sum(isGood);
totalCount = length([s.inputGain]);
fprintf('Number of good weights: %d out of %d (%.2f%%)\n', goodCount, totalCount, (goodCount/totalCount)*100);

%%

% Define the range of input and interference gains
inputGainMin = 4;
inputGainMax = max(inputGaindB);
interferenceGainMin = -70;
interferenceGainMax = max(interferenceGaindB);

% Number of levels (bins)
numInputLevels = 4;
numInterferenceLevels = 15;

% Define bin edges for input and interference gain
inputEdges = linspace(inputGainMin, inputGainMax, numInputLevels + 1);
interferenceEdges = linspace(interferenceGainMin, interferenceGainMax, numInterferenceLevels + 1);

% 2D histogram: Count weights in each combination of input and interference levels
counts = histcounts2(inputGaindB, interferenceGaindB, inputEdges, interferenceEdges);

% Create a heatmap
figure;
imagesc(interferenceEdges(1:end-1), inputEdges(1:end-1), counts);
colorbar;
title('Distribution of Interference Gain Levels Across Input Gain Levels');
xlabel('Interference Gain (dB)');
ylabel('Input Gain (dB)');
set(gca, 'YDir', 'normal'); % Flip Y-axis to match logical layout
colormap('hot');

% Display counts as text in the heatmap
for i = 1:size(counts, 1)
    for j = 1:size(counts, 2)
        if counts(i, j) > 100
            text(interferenceEdges(j), inputEdges(i), num2str(counts(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', 'b', 'FontSize', 8);
        end
    end
end
