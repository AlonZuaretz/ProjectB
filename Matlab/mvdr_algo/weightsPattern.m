function P = weightsPattern(w, steeringVecMat)
    AF = zeros(size(steeringVecMat,1),1); % Pre-allocate MUSIC spectrum
    for i = 1:size(steeringVecMat,1)
        a = steeringVecMat(i,:)';
        AF(i) = conj(w) * a;
    end
    P = 4 * abs(AF).^2 ./ max(abs(AF).^2);
end