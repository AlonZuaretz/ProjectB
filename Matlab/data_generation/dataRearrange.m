clear
files = dir('C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV6\raw_data');
% files = dir('C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\raw_data');
files = files([files.isdir] ~= 1);

N = length(files);
load([files(1).folder,'\', files(1).name])
numReps = length(NNdata);

Xw = zeros(N*numReps,4,1);
Yw = zeros(N*numReps,4,1);

XRd = zeros(N*numReps,4,4);
XR = zeros(N*numReps,4,4);
YR = zeros(N*numReps,4,4);

Xdoa = zeros(N*numReps,4);
Ydoa = zeros(N*numReps,4);

params = struct('N', [], 'SNR', [], 'SIR', [], 'numInt', [], 'intMode', [], 'inputMode', [], 'inputAngle', [] ...
    ,'interferenceAngle', [], 'intBW', [], 'inputBW', [], 'seed', []);


for f = 1:N
    load([files(f).folder,'\', files(f).name])
	for i = 1:numReps
        globIterNum = (f-1)*numReps + i;
		Yw(globIterNum, :, :) = NNdata(i).wMVDR;
		XRd(globIterNum, :, :) = NNdata(i).RMPDR_distorted;
        XR(globIterNum, :, :) = NNdata(i).RMPDR;
		YR(globIterNum, :, :) = NNdata(i).RMVDR;
		Ydoa(globIterNum, :, :) = NNdata(i).trueSteeringVector;
		params(globIterNum) = NNdata(i).params;
	end
end

save('C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV6\dataForPython.mat' ...
    , 'Xw', 'XR', 'XRd', 'Yw', 'YR', 'Xdoa', 'Ydoa', 'params')
% save('C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\dataForPython.mat' ...
%     , 'Xw', 'XR', 'XRd', 'Yw', 'YR', 'Ydoa', 'params')
