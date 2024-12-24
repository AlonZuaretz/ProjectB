clear
set(0, 'DefaultFigureWindowStyle', 'docked');
addpath('Matlab\mvdr_algo\')

% globDir = 'C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV6';
globDir = 'C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV7';
paramsDir = '';
dataDir = '\raw_data';
saveParamsDir = [globDir, paramsDir];
saveDataDir = [globDir, dataDir];
saveFlag = true;

if saveFlag
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
thetaRange = [-60, 60]; % [deg]
thetaDistRange = [40, 90]; % [deg]

% changing parameters:
snrRange = [20, 40];  % vector of SNR values
sirRange = [-30, -5];  % vector of SIR values
numInt = 1; % vector of number of interferences

intMode = ["filtNoise", "CW"];
inputMode = ["filtNoise", "CW"];

% how many times to randomize:
randAnglesNum = 10000;
randReps = 10;


N1 = length(numInt);
N2 = length(intMode);
N3 = length(inputMode);
N4 = randAnglesNum;
N5 = randReps;

Nreps = N1 * N2 * N3 * N4 * N5

% Additional parameters to save:
addParams.dtheta = dtheta;
addParams.thetaRange = thetaRange;
addParams.thetaDistRange = thetaDistRange;
addParams.snrRange = snrRange;
addParams.sirRange = sirRange;
addParams.numInt = numInt;
addParams.intMode = intMode;
addParams.inputMode = inputMode;
addParams.randAnglesNum = randAnglesNum;
addParams.randReps = randReps;
addParams.Nreps = Nreps;

if saveFlag
    save([saveParamsDir, '\globalParams.mat'], 'params', 'addParams');
end
%% Generation Flow:

h = waitbar(0, 'Please wait... Progress loading');

for i1 = 1:N1 % iterate through number of interferences
    nInt = numInt(i1);
    params.numInt = nInt;

    for i2 = 1:N2 % iterate through modes of interferences
        intM = intMode(i2);
        % params.intMode is updated at i7 for loop

        for i3 = 1:N3 % iterate through modes of input
            inputM = inputMode(i3);
            % params.inputMode is updated at i7 for loop

            if strcmp(intM, "CW") && strcmp(inputM, "CW")
                continue;
            end

            for i4 = 1:N4 % iterate through a set of random angles
                % generate avoidance angles:
                interferenceAngle = randi(thetaRange, numInt, 1);
                interferenceAngle = [interferenceAngle.' ; zeros(1, numInt)];
                
                % generate input angle:
                valid = false;  % Initialization of the validity check
                while ~valid
                    inputAngle = randi(thetaRange, 1);
                    distances = abs(inputAngle - interferenceAngle(1,:)); 
                    if distances >= thetaDistRange(1) && distances <= thetaDistRange(2)  % Check if all distances are at least c
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
                NNdata = struct();
                for i5 = 1:N5
                    currIter = calcIter(i1, i2, i3, i4, i5 ,N2 ,N3 ,N4 ,N5);
                    if ~mod(currIter, 1e4)
                        disp(currIter)
                    end
                    % loading bar:
                    waitbar(currIter / Nreps, h, sprintf('Progress: %d%%', floor(currIter / Nreps * 100)));
                    struct = mainIterator(params, mvdrBeamFormer, nInt, snrRange, sirRange, inputAngle, interferenceAngle, intSteeringVector, inputSteeringVector, intM, inputM, currIter);
                    NNdata(i5).RMVDR = struct.RMVDR;
                    NNdata(i5).wMVDR = struct.wMVDR;
                    NNdata(i5).RMPDR_distorted = struct.RMPDR_distorted;
                    NNdata(i5).RMPDR = struct.RMPDR;
                    NNdata(i5).trueSteeringVector = struct.trueSteeringVector;
                    NNdata(i5).params = struct.params;
                end
                if saveFlag
                    C = {[saveDataDir, '\run'], num2str(i1), num2str(i2), num2str(i3), num2str(i4)};
                    filename = strjoin(C, '_');
                    save([filename, '.mat'], 'NNdata');
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

function globalIter = calcIter(i1, i2, i3, i4, i5, N2,N3,N4,N5)
    globalIter = 1 + (i1-1) * N2 * N3 * N4 * N5  ...
                    + (i2-1) * N3 * N4 * N5  ...
                    + (i3-1) * N4 * N5  ...
                    + (i4-1) * N5  ...
                    + (i5-1);
end

































