% Script to generate the exact signals used to achieve samples for training

clear
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\raw_data\run_1_1_1.mat");
relData = NNdata(1);

%%
% angular resolution:
dtheta = addParams.dtheta; % [deg]
thetaMin = addParams.thetaMin; % [deg]
thetaMax = addParams.thetaMax; % [deg]
thetaDist = addParams.thetaDist; % [deg]

% changing parameters:
snrRange = addParams.snrRange;  % vector of SNR values
sirRange = addParams.sirRange;  % vector of SIR values
numInt = addParams.numInt; % vector of number of interferences

intMode = addParams.intMode;
inputMode = addParams.inputMode;

nInt = relData.params.numInt;
intM = relData.params.intMode;
inputM = relData.params.inputMode;
interferenceAngle = relData.params.interferenceAngle;
inputAngle = relData.params.inputAngle;
seed = relData.params.seed;

params.numInt = nInt;
params.interferenceAngle = interferenceAngle;
params.inputAngle = inputAngle;

mvdrBeamFormer = MyMVDRBeamFormer(params);
inputSteeringVector = mvdrBeamFormer.SteeringVector;
intSteeringVector = mvdrBeamFormer.calcSteeringVec(interferenceAngle);


NNdata = struct();
currIter = seed;
struct = mainIterator(params, mvdrBeamFormer, nInt, snrRange, sirRange, inputAngle, interferenceAngle, intSteeringVector, inputSteeringVector, intM, inputM, currIter);
