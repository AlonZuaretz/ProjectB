% Given a data file extracted from python with relevant parameters. This
% script provides a full restoration of the signals that were used to
% create the MVDR, MPDR, wMVDR, wMPDR.
% 
clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1\lowSIR\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1\lowSIR\netV2_results\data_LA4M1.mat")

%%
SIRval = 0;
SIRidxs = find([pythonParams.SIR] == SIRval);

relIdx = SIRidxs(1);
fsBB = params.fsBB;

relStruct = pythonParams(relIdx);
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

relXR = double(squeeze(XR(relIdx,:,:)));
relYR_true = double(squeeze(YR_true(relIdx,:,:)));

mvdrBeamFormer.CovarianceMatrix = relYR_true;
wMVDR = mvdrBeamFormer.calcWeights();


relYR1 = double(squeeze(YR(relIdx,:,:)));
mvdrBeamFormer.CovarianceMatrix = relYR1;
wM1 = mvdrBeamFormer.calcWeights();

% load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1\lowSIR\netV2_results\data_LA2M2.mat")
% relYR = double(squeeze(YR(relIdx,:,:)));
% mvdrBeamFormer.CovarianceMatrix = relYR;
% wM2 = mvdrBeamFormer.calcWeights();

load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1\lowSIR\netV2_results\data_LA4M1_finetune.mat")
relYR3 = double(squeeze(YR(relIdx,:,:)));
mvdrBeamFormer.CovarianceMatrix = relYR3;
wM3 = mvdrBeamFormer.calcWeights();


figure;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

hold on;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wM1,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');

% pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wM2,'Type','directivity',...
%     'PropagationSpeed',params.c,...
%     'CoordinateSystem','rectangular');

pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wM3,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');

xline([inputAngle(1), interferenceAngle(1)], 'LineWidth', 1.5)
title(['SIR = ', num2str(SIRval), '[dB]'])
legend('wMVDR','wM1','wM1_finetuned', 'Location','southeast')




