function paramsOut = genParams(varargin)
    % Inputs:
    % optional: varargin{1} - SNR
    %           varargin{2} - SIR
    % outputs:
    % struct paramsOut. contains the fields below


    % Fix parameter values:
    M = 4; % Number of array elements
    carrierFreq = 28e9; % [ Hz ]
    effCarrierFreq = 0;
    c = physconst('LightSpeed'); %  light speed in [ m/s ]
    lambda = c/carrierFreq; %  wave length in [ m ]
    d = lambda/2; % distance between array elements in [ m ]
    fs = 125e6; % sampling frequency
    T = 1e-6; % total signal duration
    t = (0:1/fs:T-1/fs).';
    N = size(t,1);
    intMode = 'noise'; % interference mode (noise or correlated)
    ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
    load('Matlab\data_generation\element_gain_mismatch.mat', 'element_gain_mismatch');
    load('Matlab\data_generation\element_ang_mismatch.mat', 'element_ang_mismatch');

    
    numInt = 1;
    SNR = 20; %dB
    SIR = -50; %dB
    if nargin == 1
        SNR = varargin{1};
    end
    if nargin == 2
        SIR = varargin{2};
    end
    % calculate SINR:
    SNR_lin = 10^(SNR/10);
    SIR_lin = 10^(SIR/10);
    SINR_lin = (1/SNR_lin + 1/SIR_lin)^-1;
    SINR = 10*log10(SINR_lin);
    
    % angles:
    inputAngle = [10;0];
    switch numInt
        case 1
            interferenceAngle = [-10;0];
        case 2
            interferenceAngle = [[-30 ; 0] , [20;0]];
        case 3
            interferenceAngle = [[-60 ; 0] , [-30 ; 0] , [30;0]];
    end

    %%

    paramsOut.M = M;
    paramsOut.carrierFreq = carrierFreq;
    paramsOut.effCarrierFreq = effCarrierFreq;
    paramsOut.c = c;
    paramsOut.lambda = lambda;
    paramsOut.d = d;
    paramsOut.fs = fs;
    paramsOut.ula_array = ula_array;
    paramsOut.gainMatrix = element_gain_mismatch;
    paramsOut.phaseMatrix = element_ang_mismatch;
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