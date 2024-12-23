clear
files = dir('C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\raw_train');
% files = dir('C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\raw_train');
files = files(4:end);

N = length(files);
load([files(1).folder,'\', files(1).name])
numAngleReps = length(angleRepsStruct);
numReps = length(angleRepsStruct(1).RepsStruct);

Xw = zeros(N*numReps*numAngleReps,4,1);
Yw = zeros(N*numReps*numAngleReps,4,1);

XRd = zeros(N*numReps*numAngleReps,4,4);
XR = zeros(N*numReps*numAngleReps,4,4);
YR = zeros(N*numReps*numAngleReps,4,4);

Xdoa = zeros(N*numReps*numAngleReps,4);
Ydoa = zeros(N*numReps*numAngleReps,4);

params = struct('N', [], 'SNR', [], 'SIR', [], 'numInt', [], 'intMode', [], 'inputMode', [], 'inputAngle', [] ...
    ,'interferenceAngle', [], 'intBW', [], 'inputBW', [], 'seed', []);


for f = 1:N
    load([files(f).folder,'\', files(f).name])
    for j = 1:numAngleReps
        RepsStruct = angleRepsStruct(j).RepsStruct;
		for i = 1:numReps
            globIterNum = (f-1)*numReps*numAngleReps + (j-1)*numReps + i;
			% Xw(globIterNum, :, :) = RepsStruct(i).wMPDR;
			Yw(globIterNum, :, :) = RepsStruct(i).wMVDR;
			XRd(globIterNum, :, :) = RepsStruct(i).RMPDR_distorted;
            XR(globIterNum, :, :) = RepsStruct(i).RMPDR;
			YR(globIterNum, :, :) = RepsStruct(i).RMVDR;
			% Xdoa(globIterNum, :, :) = RepsStruct(i).estDOA;
			Ydoa(globIterNum, :, :) = RepsStruct(i).trueSteeringVector;
			params(globIterNum) = RepsStruct(i).params;
		end
    end
end
save('C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat' ...
    , 'Xw', 'XR', 'XRd', 'Yw', 'YR', 'Xdoa', 'Ydoa', 'params')
% save('C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat' ...
%     , 'Xw', 'XR', 'Yw', 'YR', 'Xdoa', 'Ydoa', 'params')
