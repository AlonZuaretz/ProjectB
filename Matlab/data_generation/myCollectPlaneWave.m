function y = myCollectPlaneWave(x, params, angle, steeringVec, GPflag)
    if GPflag
        thetaIdx = round(angle(1,:) + 61);
        gainVec = params.gainMatrix(thetaIdx, :).';
        phaseVec = params.phaseMatrix(thetaIdx, :).';
        steeringVec = steeringVec .* gainVec .* exp(1i * deg2rad(phaseVec));
    end

    y = x * steeringVec.' ; 
end
    
    
    