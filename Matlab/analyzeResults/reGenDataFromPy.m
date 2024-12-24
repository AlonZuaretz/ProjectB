% Given a data file extracted from python with relevant parameters. This
% script provides a full restoration of the signals that were used to
% create the MVDR, MPDR, wMVDR, wMPDR.


clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\NN_results\stage1_run_20241210_141656\test_results.mat")

%%
snrRange = addParams.snrRange;
sirRange = addParams.sirRange;

SIRval = -40;
SNRval = 25;
Nval = 2^10;
intMode = 'CW';
inputMode = 'filtNoise';

for i = 1:length(pythonParams)
    term1(i) = pythonParams(i).SIR == SIRval;
end
for i = 1:length(pythonParams)
    term2(i) = pythonParams(i).SNR == SNRval;
end
for i = 1:length(pythonParams)
    term3(i) = pythonParams(i).N >= Nval;
end
term4 = strcmp([pythonParams.intMode], intMode);
term5 = strcmp([pythonParams.inputMode], inputMode);
terms = term1 .* term2 .* term3 .* term4 .* term5;
relIdxs = find(terms);


relIdx = relIdxs(1);
fsBB = params.fsBB;
relStruct = pythonParams(relIdx);
relXR = double(squeeze(input_XRd(relIdx,:,:)));
relYR = double(squeeze(output_YR(relIdx,:,:)));
relYR_true = double(squeeze(label_YR(relIdx,:,:)));


params.SNR = double(relStruct.SNR);
params.SIR = double(relStruct.SIR);
params.numInt = double(relStruct.numInt);
params.inputAngle = double(relStruct.inputAngle);
params.interferenceAngle = double(relStruct.interferenceAngle);

inputAngle = params.inputAngle;
interferenceAngle = params.interferenceAngle;

mvdrBeamFormer = MyMVDRBeamFormer(params);
inputSteeringVector = mvdrBeamFormer.SteeringVector;
intSteeringVector = mvdrBeamFormer.calcSteeringVec(interferenceAngle);

rng(relStruct.seed)
repSNR = randi(snrRange);
repSIR = randi(sirRange);
params.N = params.Nopt;
params.T = params.Topt;
params.t = params.topt;

% create signals for optimal MVDR matrix
% calculation:
params.intMode = 'noise';
[SoA_noise, noise] = randSimSignals(params);

% Optimal reception - no G/P distortion:
GPflag = false;
interference = myCollectPlaneWave(SoA_noise, params, interferenceAngle,  intSteeringVector, GPflag);
rxInt = interference + noise;
% calculate optimal MVDR matrix and coefficients:
[RMVDR, wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);

RMVDR = RMVDR ./ max(abs(RMVDR),[], 'all');

nSamples = randi([4, 2^16]);
T = nSamples/fsBB; 
t = (0:nSamples-1).' / fsBB;

intM = relStruct.intMode;
inputM = relStruct.inputMode;
if strcmp(intM, 'filtNoise')
    intBW = randi([1,6]);
else
    intBW = [];
end
if strcmp(inputM, 'filtNoise')
    inputBW = randi([1,6]);
else
    inputBW = [];
end

params.N = nSamples;
params.T = T;
params.t = t;
params.intMode = intM;
params.inputMode = inputM;
params.intBW = intBW;
params.inputBW = inputBW;

[SoA_corr, noise, SoI] = randSimSignals(params);

% add G/P distortions in the reception
GPflag = true;
x = myCollectPlaneWave(SoI, params, inputAngle, inputSteeringVector, GPflag);
interference = myCollectPlaneWave(SoA_corr, params, interferenceAngle, intSteeringVector, GPflag);
rxSignal = x + interference + noise;
% calculate distorted MPDR matrix and coefficients:
[RMPDR, wMPDR] = mvdrBeamFormer.mvdrTrain(rxSignal);
RMPDR = RMPDR ./ max(abs(RMPDR),[], 'all');

w = mvdrBeamFormer.calcWeights(relYR);

figure;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

hold on;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMPDR,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');


pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',w,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');

xline([inputAngle(1), interferenceAngle(1)], 'LineWidth', 1.5)
title(['SIR = ', num2str(SIRval), '[dB]'])
legend('wMVDR','wMPDR', 'Reconstructed', 'Location','southeast')




