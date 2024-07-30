%% Parameters:
clear
set(0, 'DefaultFigureWindowStyle', 'docked');

params = genParams();
M = params.M;
carrierFreq = params.carrierFreq;
c = params.c;
lambda = params.lambda;
d = params.d;
fs = params.fs;
T = params.T;
t = params.t;
N = params.N;
SNR = params.SNR;
SIR = params.SIR;
SINR = params.SINR;
numInt = params.numInt;
intMode = params.intMode;
inputAngle = params.inputAngle;
interferenceAngle = params.interferenceAngle;

    
%% Simulate signals:

[SoI, SoA, noise] = simSignals(params);
signalPower = mean(abs(SoI).^2);

%% Phased Array Flow

% create the phased array
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);

% Initialize the MVDR beamformer
mvdrBeamFormer = MyMVDRBeamFormer(params);
mvdrBF = phased.MVDRBeamformer('SensorArray',ula_array,...
    'Direction',inputAngle,'OperatingFrequency',carrierFreq,...
    'WeightsOutputPort',true);
% sample signals using array:
x = collectPlaneWave(ula_array, SoI, inputAngle, carrierFreq);
interference = collectPlaneWave(ula_array, SoA, interferenceAngle, carrierFreq);


rxInt = interference + noise;
rxSignal = x + rxInt;



%% get y using beamformers:
[covMatrix , wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);
yMVDR = mvdrBeamFormer.mvdrBeamFormer(rxSignal);
mvdrBF.TrainingInputPort = true;
[yMVDRBF, wMVDRBF] = mvdrBF(rxSignal, rxInt);

% calc SNR at output
noiseTemp = abs(SoI) - abs(yMVDR);
noisePower = mean(abs(noiseTemp).^2);
outSNR = 10*log10(signalPower/noisePower);
%% plots:
figure;
plot(t, abs(yMVDR))
xlim([0, 2/carrierFreq])
legend('MVDR')

figure;
pattern(ula_array,carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
    'PropagationSpeed',c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);