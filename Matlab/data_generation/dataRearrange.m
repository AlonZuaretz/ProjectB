clear
files = dir('C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\raw_train');
files = files(3:end);

N = length(files);
load([files(1).folder,'\', files(1).name])
numReps = length(RepsStruct);
Xw = zeros(N*numReps,4,1);
Yw = zeros(N*numReps,4,1);

XR = zeros(N*numReps,4,4);
YR = zeros(N*numReps,4,4);

params = struct('N', [], 'SNR', [], 'SIR', [], 'numInt', [], 'intMode', [], 'inputMode', [], 'inputAngle', [] ...
    ,'interferenceAngle', [], 'intBW', [], 'inputBW', [], 'seed', []);


for f = 1:N
    load([files(f).folder,'\', files(f).name])
    for i = 1:numReps
        Xw((f-1)*numReps + i, :, :) = RepsStruct(i).wMPDR;
        Yw((f-1)*numReps + i, :, :) = RepsStruct(i).wMVDR;
        XR((f-1)*numReps + i, :, :) = RepsStruct(i).RMPDR;
        YR((f-1)*numReps + i, :, :) = RepsStruct(i).RMVDR;
        Xdoa((f-1)*numReps + i, :, :) = RepsStruct(i).DOA;
        Ydoa((f-1)*numReps + i, :, :) = RepsStruct(i).params.inputAngle;
        params((f-1)*numReps + i) = RepsStruct(i).params;
    end
end
save('C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\dataForPython_train.mat' ...
    , 'Xw', 'XR', 'Yw', 'YR','Xdoa', 'Ydoa', 'params')
