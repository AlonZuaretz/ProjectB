clear
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\globalParams.mat")
load("C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\NN_results\data.mat")
%%
SIRvals = addParams.SIRs;
SIRvals_str = cellstr('SIR_' + strrep(string(SIRvals), '-', 'n'));
SIRidxs = struct();


for i = 1:length(SIRvals)
    SIRidxs.(SIRvals_str{i}) = find([pythonParams.SIR] == SIRvals(i));
end


%% calculate MSE in covariance matrix for each SIR separately:
MSE = zeros(length(SIRvals),1);
for i = 1:length(SIRvals)
    relYR_true = double(YR_true(SIRidxs.(SIRvals_str{i}),:,:) );
    relYR = double(YR(SIRidxs.(SIRvals_str{i}),:,:) );
    MSE(i) = mean(abs(relYR_true - relYR).^2, 'all');
end
    
figure;
plot(SIRvals, 20*log10(MSE))
xlabel('SIR [dB]')
ylabel('MSE [dB]')
grid('on')
title('MSE vs SIR')









