classdef MyMVDRBeamFormer < handle
    % MVDRBeamformer performs beamforming using the MVDR algorithm.
    % Only the method mvdrTrain updates the object properties

    %% TODO:
    % Improve estimation of DOA:
    % Define the set of steering vectors as a class parameter and then
    % don't calculate it every time.
    % Then, use the following formula:
    % pPattern = 1./real(sum(sv'.*(Cx\sv).',2));
    % where sv is 4 x thetasNum, Cx is the covariance matrix.
    %%
    properties
        Array % Array object
        CovarianceMatrix % Covariance matrix of the array inputs
        WeightsVector
        desiredDOA % Desired direction of arrival (DOA) for the beamforming
        CarrierFreq
        SteeringVector % Array steering vector
        c % speed of light
    end
    methods
        % Constructor to initialize the MVDR Beamformer
        function obj = MyMVDRBeamFormer(params)
            desiredDOA = params.inputAngle;
            carrierFreq = params.carrierFreq;
            ulaArray = params.ula_array;

            obj.Array =  ulaArray;
            obj.CovarianceMatrix = [];
            obj.WeightsVector = [];
            obj.desiredDOA = desiredDOA;
            obj.CarrierFreq = carrierFreq;
            obj.c = physconst('LightSpeed');

            % DOA might be unknown at this stage:
            if ~isempty(desiredDOA)
                obj.SteeringVector = obj.calcSteeringVec();
            else
                obj.SteeringVector = [];
            end
        end

        %%%%%
        function varargout = mvdrTrain(obj, signal, varargin)
            % Inputs:
            % Signal - either with interferences or not.
            % varargin{1} - Diagonal loading factor

            % Calculates:
            % Covariance matrix (Normalized by the number of samples)
            % Estimates steering vector if inputAngle has not been given at
            % initalization
            % Phase weights vector

            N = size(signal,1);
            if ~isempty(varargin) % diagonal loading
                lambda = varargin{1};
                d = size(signal,2);
                R = (signal.' * conj(signal))/N + lambda * eye(d);
            else
                R = (signal.' * conj(signal))/N;
            end

            obj.CovarianceMatrix = R;
            
            if isempty(obj.SteeringVector)
                doa = obj.estimateDOA(1);
                obj.desiredDOA = doa;
                obj.SteeringVector = obj.calcSteeringVec();
            end
            
            w = obj.calcWeights();
            obj.WeightsVector = w;

            varargout{1} = R;
            varargout{2} = w;
        end
        %%%%%%

        %%%%%%
        function y = mvdrBeamFormer(obj, signal)
            % apply weights over the signal:
            y = signal * conj(obj.WeightsVector);
        end
        %%%%%

        %%%%%
        function w = calcWeights(obj, varargin)
            % varargin{1} - if empty, use obj.CovarianceMatrix,
            % if not, use varargin{1} as covariance matrix

            % calcualte weights for beamformer:

            if nargin == 2
                R = varargin{1};
            else
                R = obj.CovarianceMatrix;
            end
            invR = inv(R);

            denominator = obj.SteeringVector' * invR * obj.SteeringVector;%#ok
            nominator = invR * obj.SteeringVector;%#ok
            if denominator == 0
                error('Denominator in weight calculation is zero');
            elseif nominator == 0
                error('Nominator in weight calculation is zero')
            end
            w = nominator / denominator;
        end
        %%%%%

        %%%%%
        function steeringVec = calcSteeringVec(obj,varargin)
            if ~isempty(varargin)
                theta = varargin{1};
            else
                theta = obj.desiredDOA;
            end
            steeringVec = phased.internal.steeringvec(obj.Array.getElementPosition,...
                 obj.CarrierFreq,obj.c, theta, 0);
        end
        %%%%%

        
        %%%%%
        function doa = estimateDOA(obj, numSignals, varargin)
            % Inputs:
            % numSignals - number of signals to estimate their DOA's
            % varargin{1} - Covariance matrix (default is the trained matrix)
            % varargin{2} - theta's range (default is [-90, 90] ) 
            % if varargin{2} is a vector longer than 2 samples, then it is
            % the vector for which theta is scanned across.
            % varargin{3} - plotFlag (default is false)

            if nargin > 2 && ~isempty(varargin{1})
                R = varargin{1};
            else
                R = obj.CovarianceMatrix;
            end
            if nargin > 3 
                if length(varargin{2}) > 2
                    thetaScan = varargin{2};
                else
                    thetaScan = varargin{2}(1) : varargin{2}(2);
                end
            else
                thetaScan = -90:90;
            end
            if nargin > 4
                plotFlag = varargin{3};
            else
                plotFlag = false;
            end

            P = zeros(size(thetaScan)); 
            Rnorm = R/ max(abs(R), [], 'all');
            for i = 1:length(thetaScan)
                theta = [thetaScan(i) ; 0];
                a = obj.calcSteeringVec(theta);
                P(i) = 1 / (a' * inv(Rnorm) * a); %#ok
            end

            % find peaks of the spatial power spectrum
            [vals, locs] = findpeaks(abs(P), thetaScan, 'MinPeakDistance', 10, 'NPeaks', numSignals+1);
            [~, minValIdx] = min(vals);
            doa = locs(minValIdx);
            doa = [doa(:)' ; zeros(length(doa),1)];
            
            if plotFlag
                figure;
                plot(thetaScan, 10 * log10(abs(P)));
                xlabel('Angle (degrees)');
                ylabel('Spatial Spectrum (dB)');
                title('MVDR Spatial Spectrum');
                grid('on');
            end
        end
        %%%%%%
        
    end
end
