function NNparams = mainIterator(params, mvdrBeamFormer, nInt, snrRange, sirRange, inputAngle, interferenceAngle, intSteeringVector, inputSteeringVector, intM, inputM, currIter)

    fsBB = params.fsBB;

    seed = currIter;
    rng(seed) 

    repSNR = randi(snrRange);
    repSIR = randi(sirRange);
    params.SNR = repSNR;
    params.SIR = repSIR;
    
    %% optimal MVDR
    
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

    % calculate optimal MVDR matrix and coefficients:
    mvdrBeamFormer.SteeringVector = inputSteeringVector;
    [RMVDR, wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);

    %% MPDR
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

    % MPDR without G/P distortions
    GPflag = false;
    x = myCollectPlaneWave(SoI, params, inputAngle, inputSteeringVector, GPflag);
    interference = myCollectPlaneWave(SoA_corr, params, interferenceAngle, intSteeringVector, GPflag);
    rxSignal = x + interference + noise;
    % calculate MPDR matrix:
    [RMPDR, ~] = mvdrBeamFormer.mvdrTrain(rxSignal);

    % MPDR with G/P distortions 
    GPflag = true;
    x = myCollectPlaneWave(SoI, params, inputAngle, inputSteeringVector, GPflag);
    interference = myCollectPlaneWave(SoA_corr, params, interferenceAngle, intSteeringVector, GPflag);
    rxSignal = x + interference + noise;
    % calculate distorted MPDR matrix:
    [RMPDR_distorted, ~] = mvdrBeamFormer.mvdrTrain(rxSignal);

    % Saving procedure:
    saveParams = struct();
    saveParams.N = nSamples;
    saveParams.SNR = repSNR;
    saveParams.SIR = repSIR;
    saveParams.numInt = nInt;
    saveParams.intMode = cellstr(intM);
    saveParams.inputMode = cellstr(inputM);
    saveParams.inputAngle = inputAngle;
    saveParams.interferenceAngle = interferenceAngle;
    saveParams.intBW = intBW;
    saveParams.inputBW = inputBW;
    saveParams.seed = seed;

    NNparams.RMVDR = RMVDR;
    NNparams.wMVDR = wMVDR;
    NNparams.RMPDR_distorted = RMPDR_distorted;
    NNparams.RMPDR = RMPDR;
    NNparams.inputAngle = inputAngle;
    NNparams.trueSteeringVector = inputSteeringVector;
    NNparams.params = saveParams;
end