%%
% Parameters
N = 4; % Number of elements in the ULA
d = 0.5; % Element spacing in wavelengths
theta_scan = -90:0.1:90; % Scanning angles (degrees)
rng(NNparams.params.seed);
% Input autocorrelation matrix 
Rxx = conj(output_MPDR);

% Eigen decomposition of Rxx
[eigenvectors, eigenvalues] = eig(Rxx); 
eigenvalues = diag(eigenvalues);

% Sort eigenvalues in ascending order and get corresponding eigenvectors
[~, idx] = sort(eigenvalues, 'ascend'); 
eigenvectors = eigenvectors(:, idx);

% Determine the number of sources (assume 2 sources)
num_sources = 2; 
noise_subspace = eigenvectors(:, 1:N-num_sources); % Noise subspace

% Construct the ULA steering vector
steering_vector = @(theta) exp(-1j * 2 * pi * d * (0:N-1).' * sind(theta));

% MUSIC spectrum computation
P_music = zeros(size(theta_scan)); % Pre-allocate MUSIC spectrum
for i = 1:length(theta_scan)
    sv = steering_vector(theta_scan(i)); % Compute steering vector for angle theta_scan(i)
    P_music(i) = 1 / (sv' * (noise_subspace * noise_subspace') * sv); % MUSIC formula
end

% Normalize the spectrum
P_music = abs(P_music) / max(abs(P_music));

% Plot the MUSIC spectrum
% figure;
hold on
plot(theta_scan, 10*log10(P_music), 'LineWidth', 2);
grid on;
xlabel('Angle (degrees)');
ylabel('Power (dB)');
title('MUSIC DOA Estimation');
