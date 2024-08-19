clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\globalParams_train.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\netV3_results\data.mat")

%%
SIRval = -50;
SNRval = 20;
Nval = 2^10;
intMode = 'filtNoise';
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
relStruct = pythonParams(relIdx);
inputAngle = double(relStruct.inputAngle);
interferenceAngle = double(relStruct.interferenceAngle);

relXw = double(squeeze(Xw(relIdx,:))).';
relYw = double(squeeze(Yw(relIdx,:))).';
relYw_true = double(squeeze(Yw_true(relIdx,:))).';


figure;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',relYw_true,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

hold on;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',relXw,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');


pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',relYw,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');

xline([inputAngle(1), interferenceAngle(1)], 'LineWidth', 1.5)
title(['SIR = ', num2str(SIRval), '[dB]'])
legend('wMVDR','wMPDR', 'Reconstructed', 'Location','southeast')




