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

%% iterate over SIR's:
SIRs = -100:100;
outSNRs = zeros(size(SIRs));
signalPower = mean(abs(SoI).^2);

for i = 1:length(SIRs)
    sir = SIRs(i);
    params.SIR = sir;

    % Train mvdr using noise interferences:
    params.intMode = 'noise';
    [~, SoA_noise, ~] = simSignals(params);
    interference = collectPlaneWave(ula_array, SoA_noise, interferenceAngle, carrierFreq);
    rxInt = interference + noise;
    % rxSignal = x + rxInt;
    [RMVDR, wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt); %% MPDR (rxSignal) or MVDR(rInt)
    
    % Apply mvdr beamformer on correlated interferences
    params.intMode = 'correlated';
    [~, SoA_corr, ~] = simSignals(params);
    interference = collectPlaneWave(ula_array, SoA_corr, interferenceAngle, carrierFreq);
    rxInt =  interference + noise;
    rxSignal = x + rxInt;
    yMVDR = mvdrBeamFormer.mvdrBeamFormer(rxSignal);

    % calc SNR at output:

    % only use when interference is noise:
    % if interference is not noise. then the spectrum of Y contains the
    % interferences at the carrier frequency and the result is only an
    % estimation.
    % freqIdx = carrierFreq / (fs/N);
    % X = fft(SoI)/N;
    % Y = fft(yMVDR)/N;
    % H = X(freqIdx+1) / Y(freqIdx+1);
    % tempNoise = SoI - yMVDR / H;
    % noisePower = mean(abs(tempNoise).^2);
    % outSNRs(i) = 10*log10(signalPower/noisePower);

    % only works for MVDR: % TODO: understand why and how to fix!
    outSNRs(i) = 10*log10( abs(norm(wMVDR' * x.')^2  / (wMVDR' * RMVDR * wMVDR) ));
end

% plots:
figure;
plot(t, [abs(yMVDR), abs(SoI)])
xlim([0, 2/carrierFreq])
legend('MVDR', 'SoI')

figure;
pattern(ula_array,carrierFreq,-180:180,0,'Weights',conj(wMVDR),'Type','directivity',...
    'PropagationSpeed',c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

if length(SIRs) > 1
    figure;
    plot(SIRs, outSNRs);
    grid('on')
    xlabel('SIR [dB]')
    ylabel('SNR [dB]')
    title('output SNR vs SIR for fixed input SNR')
end
