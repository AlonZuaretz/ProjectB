classdef MyMVDRBeamFormer < handle
    % MVDRBeamformer performs beamforming using the MVDR algorithm.

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

            obj.Array =  params.ula_array;
            obj.CovarianceMatrix = zeros(obj.Array.NumElements);
            obj.WeightsVector = zeros(obj.Array.NumElements,1);
            obj.desiredDOA = desiredDOA;
            obj.CarrierFreq = carrierFreq;
            obj.c = physconst('LightSpeed');
            obj.SteeringVector = obj.calcSteeringVec();
        end

        function steeringVec = calcSteeringVec(obj,varargin)
            if ~isempty(varargin)
                theta = varargin{1};
            else
                theta = obj.desiredDOA;
            end
            
            % privHstv = phased.SteeringVector('SensorArray',obj.Array,...
            % 'PropagationSpeed',obj.c,'IncludeElementResponse',false);
            % tic
            % steeringVec = step(privHstv, obj.CarrierFreq, theta);
            % toc

            % this is 5-6 times faster than the above:
            steeringVec = phased.internal.steeringvec(obj.Array.getElementPosition,...
                obj.CarrierFreq,obj.c, theta, 0);
        end

        function varargout = mvdrTrain(obj, signal, varargin)
            N = size(signal,1);
            if ~isempty(varargin) % diagonal loading
                lambda = varargin{1};
                d = size(signal,2);
                R = (signal.' * conj(signal))/N + lambda * eye(d);
            else
                R = (signal.' * conj(signal))/N;
            end
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
            w = nominator / denominator;
            obj.WeightsVector = w;
        end
    end
end
