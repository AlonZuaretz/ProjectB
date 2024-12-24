clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\NN_results\stage1_run_20241210_141656\test_results.mat")

%%
% Filter:
term1 = zeros(size(pythonParams,2),1); term2 = zeros(size(pythonParams,2),1); term3 = zeros(size(pythonParams,2),1); term4 = zeros(size(pythonParams,2),1);
for i = 1:length(pythonParams)
    term1(i) = pythonParams(i).SIR >= -15;
end
for i = 1:length(pythonParams)
    term2(i) = pythonParams(i).SNR <= inf;
end
for i = 1:length(pythonParams)
    term3(i) = abs(double(pythonParams(i).inputAngle(1)) - double(pythonParams(i).interferenceAngle(1))) >= 40;
    term4(i) = abs(double(pythonParams(i).inputAngle(1)) - double(pythonParams(i).interferenceAngle(1))) <= 80;
end
terms = term1 .* term2 .* term3 .* term4;
relIdxs = find(terms);
pythonParams = pythonParams(relIdxs);
label_YR = label_YR(relIdxs,:,:);
label_XR = label_XR(relIdxs,:,:);
input_XRd = input_XRd(relIdxs,:,:);
output_YR = output_YR(relIdxs,:,:);
output_XR = output_XR(relIdxs,:,:);
Indexes = Indexes(relIdxs);

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
npc = 3;
carrierFreq = 28e9;
lambda = c/carrierFreq;
d = lambda/2;
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
thetaScan = -60:0.2:60;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for i = 1:length(thetaScan)
    theta = [thetaScan(i) ; 0];
    steeringVecMat(i, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end
%% Estimate Input Angle:
inputGain = zeros(length(pythonParams),1);
interferenceGain = zeros(length(pythonParams),1);
angleDiff = zeros(length(pythonParams),1);
for i = 1:length(pythonParams)
    pyParams = pythonParams(i);
    inputAngle = double(pyParams.inputAngle(1));
    interferenceAngle = double(pyParams.interferenceAngle(1));
    R = squeeze(output_XR(i,:,:));

    % Estimate angle of input:
    P_music = musicSpectEst(R, steeringVecMat); 
    [~, locs] = findpeaks(log10(P_music), thetaScan, 'SortStr', 'ascend');
    if length(locs) <2
        inputGain(i) = NaN;
        interferenceGain(i) = NaN;
        continue;
    end
    estAngle = locs(1);

    % Calculate weights:
    [w, R_pc] = pc_beamformer(R, npc, M, estAngle);

    % Calculate Weights pattern at inputAngle and interferenceAngle:
    wp = weightsPattern(w, steeringVecMat);
    inputSteeringVec = steeringVecMat(thetaScan==inputAngle,:);
    interferenceSteeringVec = steeringVecMat(thetaScan==interferenceAngle,:);
    
    angleDiff(i) = abs(double(pyParams.inputAngle(1)) - double(pyParams.interferenceAngle(1)));
    inputGain(i) = wp(thetaScan==inputAngle);
    interferenceGain(i) = wp(thetaScan==interferenceAngle);
end

figure('Name', 'npc=3, MVDR');
subplot(2,1,1)
plot(10*log10(interferenceGain))
title('interference gain')
subplot(2,1,2)
plot(10*log10(inputGain))
title('input gain')


% % Combine angleDiff and inputGain into a single matrix
% data = [angleDiff(:), inputGain(:)];
% 
% % Find unique angleDiff values and compute their average inputGain
% [uniqueAngleDiff, ~, idx] = unique(data(:, 1));       % Unique angleDiff values
% avgInputGain = accumarray(idx, data(:, 2), [], @mean); % Average inputGain for duplicates
% 
% % Define interpolation grid for angleDiff
% angleDiffGrid = linspace(40, 80, 100);
% 
% % Interpolate averaged inputGain onto a regular grid
% inputGainInterpolated = interp1(uniqueAngleDiff, avgInputGain, angleDiffGrid, 'linear');
% 
% % Plot the averaged and interpolated data
% figure;
% plot(angleDiffGrid, inputGainInterpolated, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
% xlabel('Angle Difference (degrees)', 'FontSize', 12);
% ylabel('Average Input Gain (dB)', 'FontSize', 12);
% title('Average Input Gain vs Angle Difference', 'FontSize', 14);
% grid on;

