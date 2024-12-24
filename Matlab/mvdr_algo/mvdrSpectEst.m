function P = mvdrSpectEst(R, steeringVecMat)
    P = zeros(size(steeringVecMat,1),1); % Pre-allocate MUSIC spectrum
    Rinv = inv(R);
    for i = 1:size(steeringVecMat,1)
        a = steeringVecMat(i,:).';
        P(i) = 1 / (a' * Rinv * a); %#ok
    end
    P = abs(P) ./ max(abs(P));
end