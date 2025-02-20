function P = inverseDistortMvdrSpectEst(R, thetaScan, steeringVecMat, gainMat, phaseMat)
    P = zeros(size(steeringVecMat,1),1);
    Rinv = inv(R);
    for i = 1:size(steeringVecMat,1)
        thetaIdx = round(thetaScan(i)*1 + 61);
        gainVec = gainMat(thetaIdx, :).';
        phaseVec = phaseMat(thetaIdx, :).';

        a = steeringVecMat(i,:).';
        a = a .* (gainVec.^-1) .* (exp(-1i * deg2rad(phaseVec)));
        P(i) = 1 / (a' * Rinv * a); %#ok
    end
    P = abs(P) ./ max(abs(P));
end