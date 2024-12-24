clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\NN_results\stage1_run_20241210_141656\test_results_over_V6.mat")


% Filter:
term1 = zeros(size(pythonParams,2),1); term2 = zeros(size(pythonParams,2),1); term3 = zeros(size(pythonParams,2),1);
for i = 1:length(pythonParams)
    % term1(i) = pythonParams(i).SIR <= -40;
    term1(i) = pythonParams(i).SIR >= -15;
end
for i = 1:length(pythonParams)
    % term2(i) = pythonParams(i).SNR >= 40;
    term2(i) = pythonParams(i).SNR <= 20;
end
for i = 1:length(pythonParams)
    term3(i) = abs(double(pythonParams(i).inputAngle(1)) - double(pythonParams(i).interferenceAngle(1))) >= 40;
end
% terms = term1 .* term2 .* term3;
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

c = physconst('LightSpeed'); %  light speed in [ m/s ]
M = 4;
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

%%

for i = 1:length(pythonParams)
    pyParams = pythonParams(i);
    R = squeeze(output_XR(i,:,:));
    Rnorm = R/ max(abs(R), [], 'all');
    P_music = musicSpectEst(Rnorm, steeringVecMat); 
    angleDiff(i) = abs(double(pyParams.inputAngle(1)) - double(pyParams.interferenceAngle(1)));
    % Sdiff(i) = abs(double(pyParams.SNR) - double(pyParams.SIR));
    Sdiff(i) = double(pyParams.SIR);
    [~, locs] = findpeaks(log10(P_music), thetaScan, 'SortStr', 'ascend');
    if length(locs) ~= 2
        continue;
    end
    estAngle = locs(1);
    angleErrorTmp = abs(estAngle - double(pyParams.inputAngle(1)));
    angleError(i) = angleErrorTmp;
end

%%
angleDiffVec = min(angleDiff):max(angleDiff);
SdiffVec = min(Sdiff):max(Sdiff);
% Combine angleDiff, Sdiff, and angleError into a single matrix
data = [angleDiff(:), Sdiff(:), angleError(:)];

% Find unique (angleDiff, Sdiff) pairs and compute their average angleError
[uniquePairs, ~, idx] = unique(data(:, 1:2), 'rows'); % Unique rows
avgError = accumarray(idx, data(:, 3), [], @mean);    % Average of duplicates

% Extract averaged results
angleDiffUnique = uniquePairs(:, 1);
SdiffUnique = uniquePairs(:, 2);
angleErrorAvg = avgError;

% Interpolate the averaged data onto a regular grid
[X, Y] = meshgrid(angleDiffVec, SdiffVec);
Z = griddata(angleDiffUnique, SdiffUnique, angleErrorAvg, X, Y, 'linear');
% Z(Z<=0.5) = 0;
% Z(Z>0.5 & Z<=1.5) = 1;
% Z(Z>1.5 & Z<=2.5) =2;
Z(Z>10) = 10;

% Plot Heatmap
figure;
imagesc(angleDiffVec, SdiffVec, Z); % Heatmap
colorbar; % Display color scale
xlabel('Angle Difference (degrees)');
ylabel('SIR [dB]');
title('Angle Error Heatmap (Averaged Duplicates)');



% Adjust Axes
axis xy; % Maintain proper axis orientation

%%