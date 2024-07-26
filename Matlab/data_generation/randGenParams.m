function paramsOut = randGenParams()
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
    fsPB = 60e9; % sampling frequency at passband
    fsBB = 125e6; % sampling frequency at baseband
    ula_array = phased.ULA('NumElements',M,'ElementSpacing',d);
    load('Matlab\data_generation\element_gain_mismatch.mat', 'element_gain_mismatch');
    load('Matlab\data_generation\element_ang_mismatch.mat', 'element_ang_mismatch');
    Nopt = 2^16;
    Topt = Nopt/fsBB;
    topt = (0:Nopt-1).' / fsBB;

    % Changing parameters
    T = []; % total signal duration
    t = []; % signal time vector
    N = []; % signal number of samples


    intMode = ''; % interference mode (noise or correlated)
    numInt = []; % number of interfereres
    SNR = []; % signal to noise ratio in dB
    SIR = []; % signal to interference ratio in dB


    % angles:
    inputAngle = []; % angle of desired signal
    interferenceAngle = []; % angle of interfereres
    InputSteeringVector = [];
   
    %%

    paramsOut.M = M;
    paramsOut.carrierFreq = carrierFreq;
    paramsOut.effCarrierFreq = effCarrierFreq;
    paramsOut.c = c;
    paramsOut.lambda = lambda;
    paramsOut.d = d;
    paramsOut.fsPB = fsPB;
    paramsOut.fsBB = fsBB;
    paramsOut.ula_array = ula_array;
    paramsOut.gainMatrix = element_gain_mismatch;
    paramsOut.phaseMatrix = element_ang_mismatch;
    paramsOut.Topt = Topt;
    paramsOut.topt = topt;
    paramsOut.Nopt = Nopt;
    paramsOut.T = T;
    paramsOut.t = t;
    paramsOut.N = N;
    paramsOut.SNR = SNR;
    paramsOut.SIR = SIR;
    paramsOut.numInt = numInt;
    paramsOut.intMode = intMode;
    paramsOut.inputAngle = inputAngle;
    paramsOut.interferenceAngle = interferenceAngle;
    paramsOut.inputSteeringVector = InputSteeringVector;

    
end