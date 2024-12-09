clear
load('C:\Users\alonz\OneDrive - Technion\תואר\Project_B\data.mat')
%%
idx = 92;
true_MVDR = squeeze(label_YR(idx,:,:));
true_MPDR = squeeze(label_XR(idx, :,:));
distorted_MPDR = squeeze(input_XRd(idx,:,:));
output_MVDR = squeeze(output_YR(idx,:,:));
output_MPDR = squeeze(output_XR(idx,:,:));

Matrice_cells = [{true_MVDR}, {true_MPDR}, {distorted_MPDR}, {output_MVDR}, {output_MPDR}];
Matrice_cells_str = ["MVDR", "MPDR", "Distorted MPDR", "Estimated MVDR", "Estimated MPDR"];

pythonParams(idx).inputAngle
pythonParams(idx).interferenceAngle

%%
c = physconst('LightSpeed'); %  light speed in [ m/s ]
M = 4;
carrierFreq = 28e9;
lambda = c/carrierFreq;
d = lambda/2;
ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
thetaScan = -90:90;


% calculate all steering vectors:
steeringVecMat = zeros(size(thetaScan,2), M);
for i = 1:length(thetaScan)
    theta = [thetaScan(i) ; 0];
    steeringVecMat(i, :) = phased.internal.steeringvec(ula_array.getElementPosition,...
        carrierFreq,c, theta, 0);
end

%%

% R = MPDR_distorted;
% P = zeros(size(thetaScan)); 
% Rnorm = R/ max(abs(R), [], 'all');
% for i = 1:length(thetaScan)
%     a = steeringVecMat(i,:).';
%     P(i) = 1 / (a' * inv(Rnorm) * a); %#ok
% end
% 
% 
% figure;
% plot(thetaScan, 20*log10(abs(P)))
% grid on
% hold on
% 
% 
% %%
% invR = inv(R);
% theta = double(pythonParams(idx).inputAngle);
% steeringVec = phased.internal.steeringvec(ula_array.getElementPosition,...
%         carrierFreq,c, theta, 0);
% 
% denominator = steeringVec' * invR * steeringVec;%#ok
% nominator = invR * steeringVec;%#ok
% if denominator == 0
% error('Denominator in weight calculation is zero');
% elseif nominator == 0
% error('Nominator in weight calculation is zero')
% end
% w = nominator / denominator;
% 
% pattern(ula_array,carrierFreq,-90:90,0,'Weights',double(w),'Type','directivity',...
%     'PropagationSpeed',c,...
%     'CoordinateSystem','rectangular');
% xline(pythonParams(idx).inputAngle(1));
% xline(pythonParams(idx).interferenceAngle(1));
% 
% legend('Spectrum', 'Weights')


%%
f = figure('Visible','off');

for ii = 1:length(Matrice_cells)

    R = Matrice_cells{ii};
    P = zeros(size(thetaScan)); 
    Rnorm = R/ max(abs(R), [], 'all');
    for i = 1:length(thetaScan)
        a = steeringVecMat(i,:).';
        P(i) = 1 / (a' * inv(Rnorm) * a); %#ok
    end
    
    subplot(3,2,ii)
    plot(thetaScan, 20*log10(abs(P)))
    grid on
    hold on
    
        invR = inv(R);
    theta = double(pythonParams(idx).inputAngle);
    steeringVec = phased.internal.steeringvec(ula_array.getElementPosition,...
            carrierFreq,c, theta, 0);
    
    denominator = steeringVec' * invR * steeringVec;%#ok
    nominator = invR * steeringVec;%#ok
    if denominator == 0
    error('Denominator in weight calculation is zero');
    elseif nominator == 0
    error('Nominator in weight calculation is zero')
    end
    w = nominator / denominator;
    
    yyaxis right;
    pattern(ula_array,carrierFreq,-90:90,0,'Weights',double(w),'Type','directivity',...
        'PropagationSpeed',c,...
        'CoordinateSystem','rectangular');
    xline(pythonParams(idx).inputAngle(1));
    xline(pythonParams(idx).interferenceAngle(1));
    
    title(Matrice_cells_str(ii))
    legend('Spectrum', 'Weights')

end

f.Visible = 'on';