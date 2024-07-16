%% Parameters:
clear

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
SINR = params.SINR;
numInt = params.numInt;
intMode = params.intMode;
inputAngle = params.inputAngle;
interferenceAngle = params.interferenceAngle;

    
%% Simulate signals:

[SoI, SoA, noise] = simSignals(params);


%% Phased Array Flow

% create the phased array
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);

x = collectPlaneWave(ula_array, SoI, inputAngle, carrierFreq);
interference = collectPlaneWave(ula_array, SoA, interferenceAngle, carrierFreq);

rxInt = interference + noise;
rxSignal = x + rxInt;

%% Define objects:

% Define the MVDR beamformer
mvdrBeamFormer = MyMVDRBeamFormer(ula_array, inputAngle, carrierFreq);

%% get y using beamformers:
[covMatrix , wMVDR] = mvdrBeamFormer.mvdrTrain(rxSignal, rxInt);
yMVDR = mvdrBeamFormer.mvdrBeamFormer(rxSignal);

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