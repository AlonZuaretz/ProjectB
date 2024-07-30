function y = myCollectPlaneWave(x, params, angle, steeringVec, GPflag)
    if GPflag
        gainMatrix = params.gainMatrix;
        phaseMatrix = params.phaseMatrix;
        thetaIdx = round(angle(1,:) + 61);
        steeringVec = steeringVec .* gainMatrix(thetaIdx, :).' .* exp(1i * deg2rad(phaseMatrix(thetaIdx,:))).';
    end

    y = x * steeringVec.' ; 
end
    
    
    