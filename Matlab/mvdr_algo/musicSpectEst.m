function P = musicSpectEst(R, steeringVecMat)
    % Eigen decomposition of Rxx
    N = 4;
    num_sources = 2;
    [eigenvectors, eigenvalues] = eig(R); 
    eigenvalues = diag(eigenvalues);
    [~, idx] = sort(eigenvalues, 'ascend'); 
    eigenvectors = eigenvectors(:, idx);
    noise_subspace = eigenvectors(:, 1:N-num_sources); % Noise subspace 
    P = zeros(size(steeringVecMat,1),1); % Pre-allocate MUSIC spectrum
    for i = 1:size(steeringVecMat,1)
        a = steeringVecMat(i,:).';
        P(i) = 1 / (a' * (noise_subspace * noise_subspace') * a); % MUSIC formula
    end
    P = abs(P) / max(abs(P));
end