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

    
%% Simulate signals:

[SoI, SoA, noise] = simSignals(params);
signalPower = mean(abs(SoI).^2);

%% Phased Array Flow

% Initialize the MVDR beamformer
mvdrBeamFormer = MyMVDRBeamFormer(params);
mvdrBF = phased.MVDRBeamformer('SensorArray',ula_array,...
    'Direction',inputAngle,'OperatingFrequency',carrierFreq,...
    'WeightsOutputPort',true);
mvdrspatialspect = phased.MVDREstimator('SensorArray',ula_array,...
        'OperatingFrequency',carrierFreq,'ScanAngles',-90:90,...
        'DOAOutputPort',true,'NumSignals',1);

% sample signals using array:

GPflag = true;
inputSteeringVec = mvdrBeamFormer.SteeringVector;
intSteeringVec = mvdrBeamFormer.calcSteeringVec(interferenceAngle);
x = myCollectPlaneWave(SoI, params, inputAngle, inputSteeringVec, GPflag);
interference = myCollectPlaneWave(SoA, params, interferenceAngle, intSteeringVec, GPflag);

rxInt = interference + noise;
rxSignal = x + rxInt;

% Find angle based on spectrum:
[~, ang] = mvdrspatialspect(x + noise);
ang = [ang ; 0];

% The input angle has changed due to GP distortions:
mvdrBeamFormer.desiredDOA = ang;
mvdrBeamFormer.SteeringVector = mvdrBeamFormer.calcSteeringVec();

%% get y using beamformers:
[covMatrixMPDR , ~] = mvdrBeamFormer.mvdrTrain(rxSignal);
[covMatrixMVDR , wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);

yMVDR = mvdrBeamFormer.mvdrBeamFormer(rxSignal);

% mvdrBF.TrainingInputPort = true;
% [yMVDRBF, wMVDRBF] = mvdrBF(rxSignal, rxInt);

% calc SNR at output
noiseTemp = abs(SoI) - abs(yMVDR);
noisePower = mean(abs(noiseTemp).^2);
outSNR = 10*log10(signalPower/noisePower);
%% plots:

mvdrspatialspect(rxInt);
figure; plotSpectrum(mvdrspatialspect); title(['Only Interference, True angle = ' num2str(interferenceAngle(1))]);
mvdrspatialspect(x + noise);
figure; plotSpectrum(mvdrspatialspect); title(['Only Signal, True angle = ' num2str(inputAngle(1))]);
mvdrspatialspect(rxSignal);
figure; plotSpectrum(mvdrspatialspect); title('Signal + Interference')



% Plot the weights pattern:
% figure;
% plot(t, abs(yMVDR))
% xlim([0, 2/carrierFreq])
% legend('MVDR')

figure;
pattern(ula_array,carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
    'PropagationSpeed',c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

