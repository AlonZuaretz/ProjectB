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


thetaScan = -60:0.1:60;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for ii = 1:length(thetaScan)
    theta = [thetaScan(ii) ; 0];
    steeringVecMat(ii, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end


%% Simulate signals:

[SoI, SoA, noise] = simSignals(params);
signalPower = mean(abs(SoI).^2);

%% Loop through thetaScan
GPflag = true;
angleDict(1:length(thetaScan)) = struct('input', [],'distorted_output', []);

for ii = 1:length(thetaScan)
    inputAngle = thetaScan(ii);
    angleDict(ii).input = inputAngle;


    % Initialize the MVDR beamformer
    mvdrBeamFormer = MyMVDRBeamFormer(params);

    % sample signals using array:
    inputSteeringVec = mvdrBeamFormer.SteeringVector;
    x = myCollectPlaneWave(SoI, params, inputAngle, inputSteeringVec, GPflag);

    rxSignal = x;

    % spectrum:
    R = (rxSignal.' * conj(rxSignal))/N;
    
    P_music = musicSpectEst(R, steeringVecMat);

    [val, idx] = max(P_music);
    angleDict(ii).distorted_output = thetaScan(idx);
end



