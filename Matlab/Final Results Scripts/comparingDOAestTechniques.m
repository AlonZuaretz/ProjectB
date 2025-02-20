clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV7\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV7\NN_results\stage1_run_20241224_052213\test_results.mat")

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
thetaScan = -60:1:60;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for i = 1:length(thetaScan)
    theta = [thetaScan(i) ; 0];
    steeringVecMat(i, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end
%% Estimate Input Angle:
NNMusic(1:length(pythonParams)) = struct('index', [], 'angleError', [], 'angleDiff', [], 'inputAngle', ...
        [], 'interferenceAngle', [], 'SIR', [], 'SNR', [], 'N', []);
NNMvdr(1:length(pythonParams)) = struct('index', [], 'angleError', [], 'angleDiff', [], 'inputAngle', ...
        [], 'interferenceAngle', [], 'SIR', [], 'SNR', [], 'N', []);
Music(1:length(pythonParams)) = struct('index', [], 'angleError', [], 'angleDiff', [], 'inputAngle', ...
        [], 'interferenceAngle', [], 'SIR', [], 'SNR', [], 'N', []);
Mvdr(1:length(pythonParams)) = struct('index', [], 'angleError', [], 'angleDiff', [], 'inputAngle', ...
        [], 'interferenceAngle', [], 'SIR', [], 'SNR', [], 'N', []);


%%
for i = 1:length(pythonParams)
    pyParams = pythonParams(i);
    inputAngle = double(pyParams.inputAngle(1));
    interferenceAngle = double(pyParams.interferenceAngle(1));
    angleDiff= abs(inputAngle - interferenceAngle);
    RNN = squeeze(output_XR(i,:,:));
    RD = squeeze(input_XRd(i, :, :));


    % NNMusic:
    P = musicSpectEst(RNN, steeringVecMat); 
    [~, locs] = findpeaks(log10(P), thetaScan, 'SortStr', 'descend', 'MinPeakProminence', 1);
    if length(locs) <2
        NNMusic(i) = struct('index', NaN, 'angleError', NaN, 'angleDiff', NaN, 'inputAngle', ...
        NaN, 'interferenceAngle', NaN, 'SIR', NaN, 'SNR', NaN, 'N', NaN);
    else
        estAngle = locs(2);
        angleError = abs(estAngle - double(pyParams.inputAngle(1)));
        NNMusic(i) = struct('index', i, 'angleError', angleError, 'angleDiff', angleDiff, 'inputAngle', ...
            inputAngle, 'interferenceAngle', interferenceAngle, 'SIR', double(pyParams.SIR), 'SNR', double(pyParams.SNR), 'N', double(pyParams.N));
    end

    % NNMvdr:
    P = mvdrSpectEst(RNN, steeringVecMat); 
    [~, locs] = findpeaks(log10(P), thetaScan, 'SortStr', 'descend', 'MinPeakProminence', 1);
    if length(locs) <2
        NNMvdr(i) = struct('index', NaN, 'angleError', NaN, 'angleDiff', NaN, 'inputAngle', ...
        NaN, 'interferenceAngle', NaN, 'SIR', NaN, 'SNR', NaN, 'N', NaN);
    else
        estAngle = locs(2);
        angleError = abs(estAngle - double(pyParams.inputAngle(1)));
        NNMvdr(i) = struct('index', i, 'angleError', angleError, 'angleDiff', angleDiff, 'inputAngle', ...
            inputAngle, 'interferenceAngle', interferenceAngle, 'SIR', double(pyParams.SIR), 'SNR', double(pyParams.SNR), 'N', double(pyParams.N));
    end

    % Music
    P = musicSpectEst(RD, steeringVecMat); 
    [~, locs] = findpeaks(log10(P), thetaScan, 'SortStr', 'descend', 'MinPeakProminence', 1);
    if length(locs) <2
        Music(i) = struct('index', NaN, 'angleError', NaN, 'angleDiff', NaN, 'inputAngle', ...
        NaN, 'interferenceAngle', NaN, 'SIR', NaN, 'SNR', NaN, 'N', NaN);
    else
        estAngle = locs(2);
        angleError = abs(estAngle - double(pyParams.inputAngle(1)));
        Music(i) = struct('index', i, 'angleError', angleError, 'angleDiff', angleDiff, 'inputAngle', ...
            inputAngle, 'interferenceAngle', interferenceAngle, 'SIR', double(pyParams.SIR), 'SNR', double(pyParams.SNR), 'N', double(pyParams.N));
    end

    Music(i) = struct('index', i, 'angleError', angleError, 'angleDiff', angleDiff, 'inputAngle', ...
        inputAngle, 'interferenceAngle', interferenceAngle, 'SIR', double(pyParams.SIR), 'SNR', double(pyParams.SNR), 'N', double(pyParams.N));

    % Mvdr:
    P = mvdrSpectEst(RD, steeringVecMat); 
    [~, locs] = findpeaks(log10(P), thetaScan, 'SortStr', 'descend', 'MinPeakProminence', 1);
    if length(locs) <2
        Mvdr(i) = struct('index', NaN, 'angleError', NaN, 'angleDiff', NaN, 'inputAngle', ...
        NaN, 'interferenceAngle', NaN, 'SIR', NaN, 'SNR', NaN, 'N', NaN);
    else
        estAngle = locs(2);
        angleError = abs(estAngle - double(pyParams.inputAngle(1)));
        Mvdr(i) = struct('index', i, 'angleError', angleError, 'angleDiff', angleDiff, 'inputAngle', ...
            inputAngle, 'interferenceAngle', interferenceAngle, 'SIR', double(pyParams.SIR), 'SNR', double(pyParams.SNR), 'N', double(pyParams.N));
    end
    
end
resultsStructsCell = {NNMusic, NNMvdr, Music, Mvdr};

%% remove NaNs
resultsStructsCellOut = {NNMusic, NNMvdr, Music, Mvdr, InverseDistort};
for ii = 1:length(resultsStructsCell)
    s = resultsStructsCell{ii};
    validIdxs = find(~isnan([s.index]));
    s = s(validIdxs);
    [~, I] = sort([s.angleError], 'descend');
    s = s(I);
    resultsStructsCellOut{ii} = s;
end

%% plot:
figure;
for ii = 1:length(resultsStructsCellOut)
    s = resultsStructsCellOut{ii};
    x = [s(:).SIR].';
    y = [s(:).angleError].';
    if isempty(s)
        continue;
    end
    
    % Find unique angleDiff values and compute their average inputGain
    [unique_x, ~, idx] = unique(x);       % Unique angleDiff values
    avg_y = accumarray(idx, y, [], @mean);
    
    % Define interpolation grid for angleDiff
    x_grid = linspace(min(x), max(x), 100);
    
    % Interpolate averaged inputGain onto a regular grid
    y_interp = interp1(unique_x, avg_y, x_grid, 'linear');
    
    % Plot the averaged and interpolated data
    plot(x_grid, y_interp, 'LineWidth', 1.5); hold on;
end
legend('NN Music', 'NN Mvdr', 'Music', 'Mvdr')
xlabel('SIR [dB]', 'FontSize', 12);
ylabel('Angle Error [degrees]')
title('Comparison of different DOA estimation methods')
grid minor;

