clear
set(0, 'DefaultFigureWindowStyle', 'docked');
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5\NN_results\stage1_run_20241210_141656\test_results.mat")


% Filter:
term1 = zeros(size(pythonParams,2),1); term2 = zeros(size(pythonParams,2),1); term3 = zeros(size(pythonParams,2),1);
for i = 1:length(pythonParams)
    % term1(i) = pythonParams(i).SIR <= -40;
    term1(i) = pythonParams(i).SIR >= -15;
end
for i = 1:length(pythonParams)
    % term2(i) = pythonParams(i).SNR >= 40;
    term2(i) = pythonParams(i).SNR <= 20;
end
for i = 1:length(pythonParams)
    term3(i) = abs(double(pythonParams(i).inputAngle(1)) - double(pythonParams(i).interferenceAngle(1))) >= 40;
end
terms = term1 .* term2 .* term3;
relIdxs = find(terms);
pythonParams = pythonParams(relIdxs);
label_YR = label_YR(relIdxs,:,:);
label_XR = label_XR(relIdxs,:,:);
input_XRd = input_XRd(relIdxs,:,:);
output_YR = output_YR(relIdxs,:,:);
output_XR = output_XR(relIdxs,:,:);
Indexes = Indexes(relIdxs);

% sort by seed:
[B, I] = sort([pythonParams.seed]);
pythonParams = pythonParams(I);
label_YR = label_YR(I,:,:);
label_XR = label_XR(I,:,:);
input_XRd = input_XRd(I,:,:);
output_YR = output_YR(I,:,:);
output_XR = output_XR(I,:,:);
Indexes = Indexes(I);

c = physconst('LightSpeed'); %  light speed in [ m/s ]
M = 4;
carrierFreq = 28e9;
lambda = c/carrierFreq;
d = lambda/2;
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
thetaScan = -60:60;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for i = 1:length(thetaScan)
    theta = [thetaScan(i) ; 0];
    steeringVecMat(i, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end

%%
idx = 5;
true_MVDR = squeeze(label_YR(idx,:,:));
true_MPDR = squeeze(label_XR(idx, :,:));
distorted_MPDR = squeeze(input_XRd(idx,:,:));
output_MVDR = squeeze(output_YR(idx,:,:));
output_MPDR = squeeze(output_XR(idx,:,:));
pyParams = pythonParams(idx);

Matrice_cells = [{distorted_MPDR}, {distorted_MPDR}, {true_MVDR}, {true_MPDR}, {output_MVDR}, {output_MPDR}];
Matrice_cells_str = ["Distorted MPDR", "distorted_MPDR", "MVDR", "MPDR", "Estimated MVDR", "Estimated MPDR"];

disp("input Angle = " + num2str(pyParams.inputAngle(1)))
disp("interference Angle = " + num2str(pyParams.interferenceAngle(1)))
disp("input mode: " + pyParams.inputMode);
disp("interference mode: " + pyParams.intMode);
disp("SNR = " + num2str(pythonParams(idx).SNR))
disp("SIR = " + num2str(pythonParams(idx).SIR))
disp("# Samples = " + num2str(pythonParams(idx).N))
disp("YR MAE = " +num2str(mean(abs(output_YR - label_YR), 'all')))
disp("XR MAE = " +num2str(mean(abs(output_XR - label_XR), 'all')))
disp("seed = " +num2str(pyParams.seed))


f = figure('Visible','off');

for ii = 1:length(Matrice_cells)
    if ii == 2
        continue;
    end
    R = Matrice_cells{ii};
    Rnorm = R/ max(abs(R), [], 'all');
    P_mvdr = mvdr(Rnorm, steeringVecMat); 
    P_music = musicSpectEst(Rnorm, steeringVecMat); 
    
    
    
    subplot(3,2,ii)
    plot(thetaScan, 20*log10(abs(P_mvdr)))
    grid on
    hold on
    plot(thetaScan, 20*log10(abs(P_music)))
    
    
    xline(pythonParams(idx).inputAngle(1));
    xline(pythonParams(idx).interferenceAngle(1));
    
    title(Matrice_cells_str(ii))
    xlabel('Angle [deg]')
    
    legend('MVDR', 'MUSIC', '', '')

end

f.Visible = 'on';

%%

function P = mvdr(R, steeringVecMat)
    P = zeros(size(steeringVecMat,1)); % Pre-allocate MUSIC spectrum
    Rinv = inv(R);
    for i = 1:size(steeringVecMat,1)
        a = steeringVecMat(i,:).';
        P(i) = 1 / (a' * Rinv * a); %#ok
    end
    P = abs(P) / max(abs(P));
end
