clear
set(0, 'DefaultFigureWindowStyle', 'docked');

%% TODO:
% make sure myCollectPlaneWave is working the same as matlab's
% collectPlaneWave

% make myCollectPlaneWave work for more than 1 signal at a time

% add noise in random bandwidth to simsignals.

%%
% fixed parameters:
params = randGenParams();
M = params.M;
carrierFreq = params.carrierFreq;
c = params.c;
lambda = params.lambda;
d = params.d;
fsPB = params.fsPB;
fsBB = params.fsBB;
ula_array = params.ula_array;

% angular resolution:
dtheta = 1; % [deg]
thetaMin = -60; % [deg]
thetaMax = 60; % [deg]
thetaDist = 10; % [deg]


% changing parameters:
SNRs = 10:20; % vector of SNR values
SIRs = -20:2:0; % vector of SIR values
numInt = 1; % vector of number of interferences

intMode = ["noise", "correlated"]; % TODO : mixture of correlated and non-correlated interferences.

% how many times to randomize:
randAnglesNum = 10;
randReps = 20;

Nreps = numel(SNRs) * numel(SIRs) * numel(numInt) * numel(intMode) * randAnglesNum * randReps

% parameters that don't change R matrix:
inputAngles = [];


for i1 = 1:length(SNRs) % iterate through snr values
    snr = SNRs(i1);
    params.SNR = snr;

    for i2 = 1:length(SIRs) % iterate throgh sir values
        sir = SIRs(i2);
        params.SIR = sir;

        for i3 = 1:length(numInt) % iterate through number of interferences
            nInt = numInt(i3);
            params.numInt = nInt;

            for i4 = 1:length(intMode) % iterate through modes of interferences
                intM = intMode(i4);
                % params.intMode is updated at i6 for loop

                for i5 = 1:randAnglesNum % iterate through a set of random angles
                    
                    % generate avoidance angles:
                    interferenceAngle = randi([thetaMin, thetaMax], numInt, 1);
                    
                    % generate input angle:
                    valid = false;  % Initialization of the validity check
                    while ~valid
                        inputAngle = randi([thetaMin, thetaMax]);
                        distances = abs(inputAngle - interferenceAngle);  % Calculate the absolute distances from all M values
                        if all(distances >= thetaDist)  % Check if all distances are at least c
                            valid = true;  % Set valid to true to break the loop
                        end
                    end

                    params.interferenceAngle = interferenceAngle;
                    params.inputAngle = inputAngle;

                    mvdrBeamFormer = MyMVDRBeamFormer(params);
                    InputSteeringVector = mvdrBeamFormer.SteeringVector;
                    InterSteeringVector = mvdrBeamFormer.calcSteeringVec(interferenceAngle);
                    
                    for i6 = 1:randReps
                        %% optimal MVDR
                        params.N = params.Nopt;
                        params.T = params.Topt;
                        params.t = params.topt;

                        % create signals for optimal MVDR matrix
                        % calculation:
                        params.intMode = 'noise';
                        [~, SoA_noise, noise] = simSignals(params);
                        
                        % Optimal reception - no G/P distortion:
                        GPflag = false;
                        interference = myCollectPlaneWave(SoA_noise, params, InterSteeringVector, GPflag);
                        rxInt = interference + noise;

                        % calculate optimal MVDR matrix and coefficients:
                        [RMVDR, wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);

                        %% distorted MPDR
                        nSamples = randi([4, 2^16]);
                        T = nSamples/fsBB; 
                        t = (0:nSamples-1).' / fsBB;
                        
                        params.N = nSamples;
                        params.T = T;
                        params.t = t;
                        params.intMode = intM;

                        [SoI, SoA_corr, noise] = simSignals(params);

                        % add G/P distortions in the reception
                        GPflag = true;
                        x = myCollectPlaneWave(SoI, params, InputSteeringVector, GPflag);
                        interference = myCollectPlaneWave(SoA_corr, params, InterSteeringVector, GPflag);
                        rxSignal = x + interference + noise;

                        % calculate distorted MPDR matrix and coefficients:
                        [RMPDR, wMPDR] = mvdrBeamFormer.mvdrTrain(rxSignal);

                        % save data:
                        C = {'generatedDataV1\run', num2str(i1), num2str(i2), num2str(i3), num2str(i4), num2str(i5), num2str(i6)};
                        filename = strjoin(C, '_');
                        save([filename, '.mat'],'RMVDR', 'wMVDR', 'RMPDR', 'wMPDR', 'params');
                    end
                end
            end
        end
    end
end






































