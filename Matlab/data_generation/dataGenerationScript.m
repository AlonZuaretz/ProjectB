clear
set(0, 'DefaultFigureWindowStyle', 'docked');

%% TODO:



% SoI should be also filtered white noise %
% save bandwidth that was used

% add noise in random bandwidth to simsignals.
% save bandwidth that was used

% -1 in input angle?

% check the produced data to see if it matches the previous mvdr flow

%%
% fixed parameters:
params = randGenParams();
fsBB = params.fsBB;

% angular resolution:
dtheta = 1; % [deg]
thetaMin = -60; % [deg]
thetaMax = 60; % [deg]
thetaDist = 10; % [deg]

% changing parameters:
SNRs = 10:20; % vector of SNR values
SIRs = -20:2:0; % vector of SIR values
numInt = 2; % vector of number of interferences

intMode = ["filtNoise", "CW"]; % TODO : mixture of correlated and non-correlated interferences.
inputMode = ["filtNoise", "CW"];

% how many times to randomize:
randAnglesNum = 10;
randReps = 10;

Nreps = numel(SNRs) * numel(SIRs) * numel(numInt) * numel(intMode) * numel(inputMode) * randAnglesNum * randReps


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
                % params.intMode is updated at i7 for loop

                for i5 = 1:length(inputMode)
                    inputM = inputMode(i5);
                    % params.inputMode is updated at i7 for loop

                    for i6 = 1:randAnglesNum % iterate through a set of random angles
                        
                        % generate avoidance angles:
                        interferenceAngle = randi([thetaMin, thetaMax], numInt, 1);
                        interferenceAngle = [interferenceAngle.' ; zeros(1, numInt)];
                        
                        % generate input angle:
                        valid = false;  % Initialization of the validity check
                        while ~valid
                            inputAngle = randi([thetaMin, thetaMax], 1);
                            distances = abs(inputAngle - interferenceAngle(1,:)); 
                            if all(distances >= thetaDist)  % Check if all distances are at least c
                                valid = true;  % Set valid to true to break the loop
                            end
                        end
                        inputAngle = [inputAngle ; 0]; %#ok
                        
                        params.interferenceAngle = interferenceAngle;
                        params.inputAngle = inputAngle;
    
                        mvdrBeamFormer = MyMVDRBeamFormer(params);
                        inputSteeringVector = mvdrBeamFormer.SteeringVector;
                        intSteeringVector = mvdrBeamFormer.calcSteeringVec(interferenceAngle);
                        
                        % For each set of the previous parameters, create a
                        % struct which will hold randReps results
                        RepsStruct = struct();
                        for i7 = 1:randReps
                            output = struct();
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
                            [RMVDR, wMVDR] = mvdrBeamFormer.mvdrTrain(rxInt);
    
                            %% distorted MPDR
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
    
                            % calculate distorted MPDR matrix and coefficients:
                            [RMPDR, wMPDR] = mvdrBeamFormer.mvdrTrain(rxSignal);
    

                            RepsStruct(i7).RMVDR = RMVDR;
                            RepsStruct(i7).wMVDR = wMVDR;
                            RepsStruct(i7).RMPDR = RMPDR;
                            RepsStruct(i7).wMPDR = wMPDR;
                            RepsStruct(i7).params = params;
                            RepsStruct(i7).rxSignal = rxSignal;
                            RepsStruct(i7).rxInt = rxInt;
                        end
                            % save data:
                            C = {'generatedDataV1\run', num2str(i1), num2str(i2), num2str(i3), num2str(i4), num2str(i5), num2str(i6)};
                            filename = strjoin(C, '_');
                            save([filename, '.mat'], 'RepsStruct');
                    end
                end
            end
        end
    end
end


%%
figure;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');
axis([-90 90 -80 20]);

hold on;
pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMPDR,'Type','directivity',...
    'PropagationSpeed',params.c,...
    'CoordinateSystem','rectangular');




































