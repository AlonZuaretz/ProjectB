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

ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
mvdrBeamFormer = MyMVDRBeamFormer(ula_array, inputAngle, carrierFreq);


ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
A = zeros(M, numInt);
for i = 1:numInt
    theta = interferenceAngle(1,i);
    phIncr = 2*pi*d*carrierFreq * sin(deg2rad(theta)) / c;
            
    steeringVec = exp(-1j * phIncr * (0:M-1).' );
    x = 1;
    y = collectPlaneWave(ula_array, x, interferenceAngle(:,i), carrierFreq);
    a = (y / y(1));
A(:,i) = a;
end

A = A.' ; 

R = (A' * A)/numInt;





