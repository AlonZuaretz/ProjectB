classdef MyMVDRBeamFormer < handle
    % MVDRBeamformer performs beamforming using the MVDR algorithm.

    properties
        Array % Array object
        CovarianceMatrix % Covariance matrix of the array inputs
        WeightsVector
        DesiredDirection % Desired direction of arrival (DOA) for the beamforming
        CarrierFreq
        SteeringVector % Array steering vector
    end
    methods
        % Constructor to initialize the MVDR Beamformer
        function obj = MyMVDRBeamFormer(params)
            array = params.ula_array;
            desiredDOA = params.inputAngle;
            carrierFreq = params.carrierFreq;
            obj.Array = array;
            obj.CovarianceMatrix = zeros(obj.Array.NumElements);
            obj.WeightsVector = zeros(obj.Array.NumElements,1);
            obj.DesiredDirection = desiredDOA(1);
            obj.CarrierFreq = carrierFreq;
            obj.SteeringVector = obj.calcSteeringVec();
        end

        function steeringVec = calcSteeringVec(obj,varargin)
            if ~isempty(varargin{1})
                theta = varargin{1};
            else
                theta = obj.DesiredDirection;
            end
            M = obj.Array.NumElements;
            d = obj.Array.ElementSpacing;
            c = 3e8;
            phIncr = 2*pi*d*obj.CarrierFreq * sind(theta) / c;
            steeringVec = exp(-1j * phIncr * (0:M-1).' );
        end

        function varargout = mvdrTrain(obj, signal, varargin)
            N = size(signal,1);
            if ~isempty(varargin) % diagonal loading
                lambda = varargin{1};
                d = size(signal,2);
                R = (signal' * signal)/N + lambda * eye(d);
            else
                R = (signal' * signal)/N;
            end
            obj.CovarianceMatrix = R;
            w = obj.calcWeights();
            varargout{1} = R;
            varargout{2} = w;
        end

        function y = mvdrBeamFormer(obj, signal)
            % apply weights over the signal:
            y = signal * obj.WeightsVector;
        end

        function w = calcWeights(obj)
            % calcualte weights for beamformer:
            invR = inv(obj.CovarianceMatrix);
            denominator = obj.SteeringVector' * invR * obj.SteeringVector;%#ok
            nominator = invR * obj.SteeringVector;%#ok
            if denominator == 0
                error('Denominator in weight calculation is zero');
            elseif nominator == 0
                error('Nominator in weight calculation is zero')
            end
            w = nominator / denominator;
            obj.WeightsVector = w;
        end
    end
end
