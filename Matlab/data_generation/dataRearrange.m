clear
files = dir('generatedDataV1_simple\raw');
files = files(3:end);

N = length(files);
numReps = 10;
Xw = zeros(N*numReps,4,1);
Yw = zeros(N*numReps,4,1);

XR = zeros(N*numReps,4,4);
YR = zeros(N*numReps,4,4);

params = struct('SNR', [], 'SIR', [], 'numInt', [], 'intMode', [], 'inputMode', [], 'inputAngle', [] ...
    ,'interferenceAngle', [], 'intBW', [], 'inputBW', [], 'seed', []);


for f = 1:N
    load(files(f).name)
    for i = 1:numReps
        Xw((f-1)*numReps + i, :, :) = RepsStruct(i).wMPDR;
        Yw((f-1)*numReps + i, :, :) = RepsStruct(i).wMVDR;
        XR((f-1)*numReps + i, :, :) = RepsStruct(i).RMPDR;
        YR((f-1)*numReps + i, :, :) = RepsStruct(i).RMVDR;
        params((f-1)*numReps + i) = RepsStruct(i).params;
    end
end
save('generatedDataV1_simple\dataForPython.mat', 'Xw', 'XR', 'Yw', 'YR', 'params')
