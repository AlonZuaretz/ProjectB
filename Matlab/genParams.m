function paramsOut = genParams(varargin)

    % Fix parameter values:
    
    M = 4; % Number of array elements
    carrierFreq = 1e4; % [ Hz ]
    c = 3e8; %  light speed in [ m/s ]
    lambda = c/carrierFreq; %  wave length in [ m ]
    d = lambda/2; % distance between array elements in [ m ]
    fs = 1e6;
    T = 2e-3;
    t = (0:1/fs:T-1/fs).';
    N = size(t,1);
    SNR = 10; %dB
    SIR = -5; %dB
    SINR = (1/SNR + 1/SIR)^-1;
    numInt = 2;
    intMode = 'noise';

    if nargin == 1
        SNR = varargin{1};
    end
    if nargin == 2
        SINR = varargin{2};
    end
    
    inputAngle = [0;0];
    switch numInt
        case 1
            interferenceAngle = [20;0];
        case 2
            interferenceAngle = [[-30 ; 0] , [20;0]];
        case 3
            interferenceAngle = [[-60 ; 0] , [-30 ; 0] , [30;0]];
    end



    paramsOut.M = M;
    paramsOut.carrierFreq = carrierFreq;
    paramsOut.c = c;
    paramsOut.lambda = lambda;
    paramsOut.d = d;
    paramsOut.fs = fs;
    paramsOut.T = T;
    paramsOut.t = t;
    paramsOut.N = N;
    paramsOut.SNR = SNR;
    paramsOut.SIR = SIR;
    paramsOut.SINR = SINR;
    paramsOut.numInt = numInt;
    paramsOut.intMode = intMode;
    paramsOut.inputAngle = inputAngle;
    paramsOut.interferenceAngle = interferenceAngle;

    
end