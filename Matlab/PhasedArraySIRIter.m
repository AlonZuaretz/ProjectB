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


% create the phased array amd mvdr object
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
mvdrBeamFormer = MyMVDRBeamFormer(ula_array, inputAngle, carrierFreq);

% obtain fixed SoI and noise for the entire run:
[SoI, ~, noise] = simSignals(params);

% sample the signal and interference
x = collectPlaneWave(ula_array, SoI, inputAngle, carrierFreq);

%% iterate over SNR's:
inSNRs = inf;
outSNRs = zeros(size(inSNRs));
signalPower = mean(abs(SoI).^2);

for i = 1:length(inSNRs)
    snr = inSNRs(i);
    params.SIR = snr;

    % Train mvdr using noise interferences:
    params.intMode = 'noise';
    [~, SoA_noise, ~] = simSignals(params);
    interference = collectPlaneWave(ula_array, SoA_noise, interferenceAngle, carrierFreq);
    rxInt = interference + noise;
    [~, wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);
    
    % Apply mvdr beamformer on correlated interferences
    params.intMode = 'correlated';
    [~, SoA_corr, ~] = simSignals(params);
    interference = collectPlaneWave(ula_array, SoA_corr, interferenceAngle, carrierFreq);
    rxInt = interference + noise;
    rxSignal = x + rxInt;
    yMVDR = mvdrBeamFormer.mvdrBeamFormer(rxSignal);

    % calc SNR at output
    noiseTemp = abs(SoI) - abs(yMVDR);
    noisePower = mean(abs(noiseTemp).^2);
    outSNRs(i) = 10*log10(signalPower/noisePower);
end

% plots:
figure;
plot(t, abs(yMVDR))
xlim([0, 2/carrierFreq])
legend('MVDR')

figure;
pattern(ula_array,carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
    'PropagationSpeed',c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);
if length(inSNRs) > 1
    figure;
    plot(inSNRs, outSNRs);
    grid('on')
end