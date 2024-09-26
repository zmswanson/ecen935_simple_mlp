classdef activation_funcs
    methods (Static)
        function y = zms_relu(x)
            y = max(0, x);
        end

        function y = zms_sigmoid(x)
            y = 1 ./ (1 + exp(-x));
        end
    end
end