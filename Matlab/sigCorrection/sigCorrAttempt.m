%% Parameters:
clear
set(0, 'DefaultFigureWindowStyle', 'docked');

params = genParams();
ula_array = params.ula_array;
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


thetaScan = -60:0.5:60;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for i = 1:length(thetaScan)
    theta = [thetaScan(i) ; 0];
    steeringVecMat(i, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end


%% Simulate signals:

[SoI, SoA, noise] = simSignals(params);
signalPower = mean(abs(SoI).^2);

%% Phased Array Flow

% Initialize the MVDR beamformer
mvdrBeamFormer = MyMVDRBeamFormer(params);

% sample signals using array:
GPflag = true;
inputSteeringVec = mvdrBeamFormer.SteeringVector;
intSteeringVec = mvdrBeamFormer.calcSteeringVec(interferenceAngle);
x = myCollectPlaneWave(SoI, params, inputAngle, inputSteeringVec, GPflag);
interference = myCollectPlaneWave(SoA, params, interferenceAngle, intSteeringVec, GPflag);

rxInt = interference + noise;
rxSignal = x + rxInt;

% spectrum:
R = (rxSignal.' * conj(rxSignal))/N;

P_music = musicSpectEst(R, steeringVecMat);
P_mvdr = mvdrSpectEst(R, steeringVecMat);

figure;
plot(thetaScan, 10*log10(P_music)); hold on;
plot(thetaScan, 10*log10(P_mvdr));
legend('MUSIC', 'MVDR');




