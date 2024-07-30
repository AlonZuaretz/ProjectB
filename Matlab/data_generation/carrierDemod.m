function xBB = carrierDemod(xPB, params)
    carrierFreq = params.carrierFreq;
    fsPB = params.fsPB;
    fsBB = params.fsBB;
    t = params.t;
    temp1 = xPB .* exp(-1i * 2 * pi * carrierFreq * t);
    xBB = resample(temp1, fsBB, fsPB);
end