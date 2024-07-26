function y = myCollectPlaneWave(x, params, steeringVec, GPflag)
    % TODO: make sure this is how to operate steering vector on x
    if ~GPflag
        y = sum(x .* steeringVec', 2);
    else
        gainMatrix = params.gainMatrix;
        phaseMatrix = params.phaseMatrix;
        interferenceAngle = params.interferenceAngle;
        thetaIdx = round(interferenceAngle + 61);
        steeringVec = steeringVec .* gainMatrix(thetaIdx, :).' .* exp(1i * rad2deg(phaseMatrix(thetaIdx,:))).';
        y = sum(x .* steeringVec', 2);
    end



end
    
    
    