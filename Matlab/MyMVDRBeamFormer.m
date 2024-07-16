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
        function obj = MyMVDRBeamFormer(array, desiredDOA, carrierFreq)
            obj.Array = array;
            obj.CovarianceMatrix = zeros(obj.Array.NumElements);
            obj.WeightsVector = zeros(obj.Array.NumElements,1);
            obj.DesiredDirection = desiredDOA(1);
            obj.CarrierFreq = carrierFreq;
            obj.SteeringVector = obj.calcSteeringVec();
        end

        function steeringVec = calcSteeringVec(obj)
            M = obj.Array.NumElements;
            d = obj.Array.ElementSpacing;
            theta = obj.DesiredDirection;
            c = 3e8;
            phIncr = 2*pi*d*obj.CarrierFreq / c;
            
            steeringVec = exp(-1j * phIncr * (0:M-1).' * sin(deg2rad(theta)));
        end

        function varargout = mvdrTrain(obj, interference)
            R = interference' * interference;
            obj.CovarianceMatrix = R;
            w = obj.calcWeights();
            varargout{1} = R;
            varargout{2} = w;
        end

        function y = mvdrBeamFormer(obj, signal)
            % apply weights over the signal:
            y = signal * conj(obj.WeightsVector);
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
            w = conj(nominator / denominator); % don't know why
            obj.WeightsVector = w;
        end


    end
end
