function varargout = randSimSignals(params)

    % Outputs are ordered as:
    % varargout{1} = SoA
    % varargout{2} = noise
    % varargout{3} = SoI
    
    M = params.M;
    effCarrierFreq = params.effCarrierFreq;
    t = params.t;
    N = params.N;
    SNR = params.SNR;
    SIR = params.SIR;
    numInt = params.numInt;
    intMode = params.intMode;
    inputMode = params.inputMode;
    inputPower = params.inputPower;
    intBW = params.intBW;
    inputBW = params.inputBW;
    
    
    % simulate signal of interest:
    if nargout == 3
        if strcmp(inputMode, 'CW')
            SoI = cos(2*pi*effCarrierFreq*t);
        elseif strcmp(inputMode, 'filtNoise')
            noise = randn(N,1) + 1i*randn(N,1);
            lpFilter = params.lpFilter{inputBW};
            filteredNoise = filter(lpFilter, noise);
            currentPower = sum(abs(filteredNoise).^2) / N ;
            scalingFactor = sqrt(inputPower / currentPower);
            SoI =  filteredNoise * scalingFactor;
        end
        varargout{3} = SoI;
    end
    

    % simulate noise and interferences with desired SNR and SINR
    % SINR = Ps / (Pint + Pnoise)
    % SNR = Ps / Pnoise

    SNRlin = 10^(SNR/10);
    SIRlin = 10^(SIR/10);
    
    % signalPower = (1/N) * sum(abs(SoI).^2); % always supposed to be equal to inputPower
    intPower = inputPower/SIRlin;
    noisePower = inputPower/SNRlin;
    
    % simulate signals of avoidance:
    SoA = zeros(size(t,1), numInt);
    if strcmp(intMode, 'noise')
        sigma = sqrt(intPower / numInt);
        SoA = (sigma/sqrt(2)) * (randn(size(SoA)) + 1i*randn(size(SoA)));

    elseif strcmp(intMode, 'CW')
        % not true when effCarrierFreq = 0
        % randPhase = -pi + 2*pi*rand(1,numInt);
        % A = sqrt((2 * intPower) / numInt^2);
        % SoA =  A * cos(2*pi*(effCarrierFreq)*t + randPhase);

        % when effCarrierFreq = 0 :
        A = sqrt(intPower) / numInt;
        SoA =  A * cos(2*pi*(effCarrierFreq)*t);
        
    elseif strcmp(intMode, 'filtNoise')
        for i = 1:numInt
            noise = randn(N,1) + 1i*randn(N,1);
            lpFilter = params.lpFilter{intBW};
            filteredNoise = filter(lpFilter, noise);
            currentPower = sum(abs(filteredNoise).^2) / N ;
            scalingFactor = sqrt(intPower / (currentPower * numInt^2));
            SoA(:,i) =  filteredNoise * scalingFactor;
        end             
    end
    varargout{1} = SoA;

    % simulate noise:
    if nargout >= 2
        noise = sqrt(noisePower/2)*(randn(N, M)+1i*randn(N, M));
        varargout{2} = noise;
    end

end