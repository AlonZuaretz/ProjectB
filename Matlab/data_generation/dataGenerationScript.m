clear
set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('Matlab\mvdr_algo\')

globDir = 'C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4';
paramsDir = '\raw_train';
dataDir = '\raw_train';
saveParamsDir = [globDir, paramsDir];
saveDataDir = [globDir, dataDir];

% Check if the folder exists
if ~exist(saveParamsDir, 'dir')
    mkdir(saveParamsDir);
    fprintf('Folder "%s" was created.\n', saveParamsDir);
else
    % Folder already exists
    fprintf('Folder "%s" already exists.\n', saveParamsDir);
end

% Check if the folder exists
if ~exist(saveDataDir, 'dir')
    mkdir(saveDataDir);
    fprintf('Folder "%s" was created.\n', saveDataDir);
else
    % Folder already exists
    fprintf('Folder "%s" already exists.\n', saveDataDir);
end

%% TODO:

%%

% fixed parameters:
params = randGenParams();
fsBB = params.fsBB;
ula_array = params.ula_array;
carrierFreq = params.carrierFreq;

% angular resolution:
dtheta = 1; % [deg]
thetaMin = -60; % [deg]
thetaMax = 60; % [deg]
thetaDist = 10; % [deg]

% changing parameters:
SNRs = 20:5:50;  % vector of SNR values
SIRs = -50:5:-10;  % vector of SIR values
numInt = 1; % vector of number of interferences

intMode = ["filtNoise", "CW"]; % TODO : mixture of correlated and non-correlated interferences.
inputMode = ["filtNoise", "CW"];

% how many times to randomize:
randAnglesNum = 50;
randReps = 20;

N1 = length(SNRs);
N2 = length(SIRs);
N3 = length(numInt);
N4 = length(intMode);
N5 = length(inputMode);
N6 = randAnglesNum;
N7 = randReps;

Nreps = N1 * N2 * N3 * N4 * N5 * N6 * N7

% Additional parameters to save:
addParams.dtheta = dtheta;
addParams.thetaMin = thetaMin;
addParams.thetaMax = thetaMax;
addParams.thetaDist = thetaDist;
addParams.SNRs = SNRs;
addParams.SIRs = SIRs;
addParams.numInt = numInt;
addParams.intMode = intMode;
addParams.inputMode = inputMode;
addParams.randAnglesNum = randAnglesNum;
addParams.randReps = randReps;
addParams.Nreps = Nreps;

save([saveParamsDir, '\globalParams.mat'], 'params', 'addParams');
%% Generation Flow:

% h = waitbar(0, 'Please wait... Progress loading');
for i1 = 1:N1 % iterate through snr values
    snr = SNRs(i1);

    for i2 = 1:N2 % iterate throgh sir values
        sir = SIRs(i2);

        for i3 = 1:N3 % iterate through number of interferences
            nInt = numInt(i3);
            params.numInt = nInt;
            mvdrspatialspect = phased.MVDREstimator('SensorArray',ula_array,...
                'OperatingFrequency',carrierFreq,'ScanAngles',-60:60,...
                'DOAOutputPort',true,'NumSignals',nInt + 1);

            for i4 = 1:N4 % iterate through modes of interferences
                intM = intMode(i4);
                % params.intMode is updated at i7 for loop

                for i5 = 1:N5 % iterate through modes of input
                    inputM = inputMode(i5);
                    % params.inputMode is updated at i7 for loop

                    angleRepsStruct = struct();
                    for i6 = 1:N6 % iterate through a set of random angles
                        
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
                        NNparams = struct();
                        for i7 = 1:N7
                            currIter = calcIter(i1, i2, i3, i4, i5, i6, i7, N2 ,N3 ,N4 ,N5 ,N6 ,N7);
                            if ~mod(currIter, 1e4)
                                disp(currIter)
                            end
                            % loading bar:
                            % waitbar(currIter / Nreps, h, sprintf('Progress: %d%%', floor(currIter / Nreps * 100)));

                            seed = currIter;
                            rng(seed) 

                            repSNR = max(min(SNRs), snr + randi([-4, 4]));
                            repSIR = min(max(SIRs), sir + randi([-4, 4]));
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

                            NNparams(i7).RMVDR = RMVDR;
                            NNparams(i7).wMVDR = wMVDR;
                            NNparams(i7).RMPDR_distorted = RMPDR_distorted;
                            NNparams(i7).RMPDR = RMPDR;
                            NNparams(i7).inputAngle = inputAngle;
                            NNparams(i7).trueSteeringVector = inputSteeringVector;
                            NNparams(i7).params = saveParams;

                        end
                            angleRepsStruct(i6).RepsStruct = NNparams;
                    end 
                        C = {[saveDataDir, '\run'], num2str(i1), num2str(i2), num2str(i3), num2str(i4), num2str(i5)};
                        filename = strjoin(C, '_');
                        save([filename, '.mat'], 'angleRepsStruct');
                end
            end
        end
    end
end


%%
% figure;
% pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMVDR,'Type','directivity',...
%     'PropagationSpeed',params.c,...
%     'CoordinateSystem','rectangular');
% axis([-90 90 -80 20]);
% 
% hold on;
% pattern(params.ula_array,params.carrierFreq,-180:180,0,'Weights',wMPDR,'Type','directivity',...
%     'PropagationSpeed',params.c,...
%     'CoordinateSystem','rectangular');


function globalIter = calcIter(i1, i2, i3, i4, i5, i6, i7, N2,N3,N4,N5,N6,N7)
    globalIter = 1 + (i1-1) * N2 * N3 * N4 * N5 * N6 * N7 ...
                    + (i2-1) * N3 * N4 * N5 * N6 * N7 ...
                    + (i3-1) * N4 * N5 * N6 * N7 ...
                    + (i4-1) * N5 * N6 * N7 ...
                    + (i5-1) * N6 * N7 ...
                    + (i6-1) * N7 ...
                    + (i7-1);


end

































