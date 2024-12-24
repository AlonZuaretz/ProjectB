clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\NN_results\stage1_run_20241210_141656\test_results_over_V6.mat")

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
terms = term1 .* term2 .* term3;
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
npc = 2;
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
pyIdx = 2118;
pyParams = pythonParams(pyIdx);
MPDR = squeeze(output_XR(pyIdx,:,:)); % MPDR
MVDR = squeeze(output_YR(pyIdx,:,:));


P_music_MPDR = musicSpectEst(MPDR, steeringVecMat); 
P_music_MVDR = musicSpectEst(MVDR, steeringVecMat); 

angleDiff = abs(double(pyParams.inputAngle(1)) - double(pyParams.interferenceAngle(1)));
SIR = double(pyParams.SIR);
[~, locs] = findpeaks(log10(P_music_MPDR), thetaScan, 'SortStr', 'ascend');

estAngle = locs(1);
angleError = abs(estAngle - double(pyParams.inputAngle(1)));

disp("input Angle = " + num2str(pyParams.inputAngle(1)))
disp("interference Angle = " + num2str(pyParams.interferenceAngle(1)))
disp("input mode: " + pyParams.inputMode);
disp("interference mode: " + pyParams.intMode);
disp("SNR = " + num2str(pyParams.SNR))
disp("SIR = " + num2str(pyParams.SIR))
disp("# Samples = " + num2str(pyParams.N))
disp("Angle estimation error = " + num2str(angleError))

% PC beamformer:
[w, R_pc] = pc_beamformer(MPDR, npc, M, estAngle);
P_pc = mvdrSpectEst(R_pc, steeringVecMat);

% Plot spectrum and weights:
figure;
hold on;
plot(thetaScan, 10*log10(P_music_MPDR));
plot(thetaScan, 10*log10(P_music_MVDR))
plot(thetaScan, 10*log10(P_pc))

yyaxis right;
pattern(params.ula_array,params.carrierFreq,-90:90,0,'Weights',w','Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');

xline(pyParams.inputAngle(1));
xline(pyParams.interferenceAngle(1));

legend('MPDR - MUSIC', 'MVDR - MUSIC', 'MVDR - PC', 'weights')





