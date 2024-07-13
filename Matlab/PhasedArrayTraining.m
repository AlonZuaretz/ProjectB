%% Parameters:
M = 4; % Number of array elements
carrierFreq = 1e4; % [ Hz ]
c = 3e8; %  light speed in [ m/s ]
lambda = c/carrierFreq; %  wave length in [ m ]
d = lambda/2; % distance between array elements in [ m ]
fs = 1e6;
T = 2e-4;
SNR = 15; %dB

%% Simulate signals:
t = (0:1/fs:T-1/fs).';

SoI = cos(2*pi*carrierFreq*t);
SoA1 = 3.7 * sin(2*pi* 1/3 *carrierFreq*t);
SoA2 = 2.4 * sin(2*pi* 5/3 *carrierFreq*t);
SoA = [SoA1, SoA2];


% angles:
inputAngle = [15;0];
interferenceAngle = [[-58 ; 0] , [67 ; 0]]; % columns are sorted by: SoA1, SoA2

% add noise with desired SNR:
signalPower = var(SoI);
desiredSNR_linear = 10^(SNR / 10);  % Convert dB to linear
noisePower = signalPower / desiredSNR_linear;
noise = sqrt(noisePower/2)*(randn(size(SoI,1), M)+1i*randn(size(SoI,1), M));

%% Phased Array Flow

% create the phased array
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);

x = collectPlaneWave(ula_array, SoI, inputAngle, carrierFreq);
interference = collectPlaneWave(ula_array, SoA, interferenceAngle, carrierFreq);

rxInt = interference + noise;
rxSignal = x + rxInt;

%% Define objects:

% Define the MVDR beamformer
mvdrbeamformer = phased.MVDRBeamformer('SensorArray',ula_array,...
    'Direction',inputAngle,'OperatingFrequency',carrierFreq,...
    'WeightsOutputPort',true);


% Define conventional beamformer for reference:
psbeamformer = phased.PhaseShiftBeamformer('SensorArray',ula_array,...
    'OperatingFrequency',carrierFreq,'Direction',inputAngle,...
    'WeightsOutputPort', true);

% Define beam scanner for DOA esimate.
spatialspectrum = phased.BeamscanEstimator('SensorArray',ula_array,...
            'OperatingFrequency',carrierFreq,'ScanAngles',-90:90);
spatialspectrum.DOAOutputPort = true;
spatialspectrum.NumSignals = 1;



%% Find AoA:
[~,ang] = spatialspectrum(rxSignal);
plotSpectrum(spatialspectrum);

%% get y using beamformers:

% MVDR:
% mvdrbeamformer.TrainingInputPort = true;
% [yMVDR, wMVDR] = mvdrbeamformer(rxSignal, rxInt);

mvdrbeamformer.TrainingInputPort = false;
[yMVDR, wMVDR] = mvdrbeamformer(rxSignal);

% Phase Shift beamformer:
[yCbf,wCbf] = psbeamformer(rxSignal);

%% plots:
figure;
plot(t, abs(yMVDR))
hold on; plot(t, abs(yCbf))
legend('MVDR', 'Cbf')

figure;
pattern(ula_array,carrierFreq,-180:180,0,'Weights',wMVDR,'Type','powerdb',...
    'PropagationSpeed',c,'Normalize',false,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

hold on;   % compare to PhaseShift
pattern(ula_array,carrierFreq,-180:180,0,'Weights',wCbf,...
    'PropagationSpeed',c,'Normalize',false,...
    'Type','powerdb','CoordinateSystem','rectangular');
hold off;
legend('MVDR','PhaseShift')