clear
load("generatedDataV1\globalParams.mat")
load("generatedDataV1\run_1_1_1_1_1_1.mat");
relData = RepsStruct(2);
fsBB = params.fsBB;


params.SNR = relData.params.SNR;
params.SIR = relData.params.SIR;
params.numInt = relData.params.numInt;
params.inputAngle = relData.params.inputAngle;
params.interferenceAngle = relData.params.interferenceAngle;

inputAngle = params.inputAngle;
interferenceAngle = params.interferenceAngle;

mvdrBeamFormer = MyMVDRBeamFormer(params);
inputSteeringVector = mvdrBeamFormer.SteeringVector;
intSteeringVector = mvdrBeamFormer.calcSteeringVec(interferenceAngle);


rng(relData.params.seed)
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


intM = relData.params.intMode;
inputM = relData.params.inputMode;

nSamples = randi([4, 2^16]);
T = nSamples/fsBB; 
t = (0:nSamples-1).' / fsBB;

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

% clearvars -except rxInt rxSignal
    