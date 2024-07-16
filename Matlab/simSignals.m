function [SoI, SoA, noise] = simSignals(params)
    rng(17);
    M = params.M;
    carrierFreq = params.carrierFreq;
    t = params.t;
    N = params.N;
    SNR = params.SNR;
    SIR = params.SIR;
    numInt = params.numInt;
    intMode = params.intMode;
    
    % simulate signal of interest:
    SoI = cos(2*pi*carrierFreq*t);

    % simulate noise and interferences with desired SNR and SINR
    % SINR = Ps / Pint + Pnoise
    % SNR = Ps / Pnoise

    SNRlin = 10^(SNR/10);
    SIRlin = 10^(SIR/10);
    
    signalPower = (1/N) * sum(abs(SoI).^2);
    intPower = signalPower/SIRlin;
    noisePower = signalPower/SNRlin;
    
    % simulate signals of avoidance:
    SoA = zeros(size(t,1), numInt);
    if strcmp(intMode, 'noise')
        sigma = sqrt(intPower / numInt);
        SoA = (sigma/sqrt(2)) * (randn(size(SoA)) + 1i*randn(size(SoA)));
    elseif strcmp(intMode, 'correlated')
        randPhase = zeros(1, numInt); %-pi + 2*pi*rand(1,numInt);
        A = sqrt((2 * intPower) / numInt^2);
        SoA =  A * cos(2*pi*carrierFreq*t + randPhase);
    end  
    
    % simulate noise:
    noise = sqrt(noisePower/2)*(randn(size(SoI,1), M)+1i*randn(size(SoI,1), M));
end