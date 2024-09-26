%{
    File: activation_funcs.m
    Author: Zachary M Swanson
    Date: 09-25-2024
    Description: This class defines activation functions for use in a multilayer perceptron (MLP).
    Usage: Access the activation functions as static methods of the class, e.g.,
              y = activation_funcs.zms_relu(x);
              y = activation_funcs.zms_sigmoid(x);
%}

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