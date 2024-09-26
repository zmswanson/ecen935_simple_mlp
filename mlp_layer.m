%{
    File: mlp_layer.m
    Author: Zachary M Swanson
    Date: 09-25-2024
    Description: This class defines a layer in a multilayer perceptron (MLP). Each layer has a
                 specified number of nodes, an activation function, weights, and biases. The class
                 provides methods to set and get the weights and biases, as well as to perform the
                 forward pass through the layer.
    Usage: Create an instance of the class with the desired parameters, e.g.,
                layer = mlp_layer(5, @activation_funcs.zms_relu, weights, biases);
            Set the weights and biases using the set_weights and set_biases methods, e.g.,
                layer.set_weights(new_weights);
                layer.set_biases(new_biases);
            Get the weights and biases using the get_weights and get_biases methods, e.g.,
                weights = layer.get_weights();
                biases = layer.get_biases();
            Perform the forward pass through the layer using the forward method, e.g.,
                [activations, outputs] = layer.forward(inputs);
%}

classdef mlp_layer
    properties
        num_nodes
        activation_function
        weights
        biases
    end

    methods
        function obj = mlp_layer(num_nodes, activation_function, weights, biases)
            arguments
                num_nodes (1, 1) {mustBeNumeric, mustBeGreaterThan(num_nodes, 0)}
                activation_function
                weights (:, :) {mustBeNumeric, mustBeInRange(weights, -1, 1)}
                biases (:, 1) {mustBeNumeric, mustBeInRange(biases, -1, 1)}
            end

            obj.num_nodes = num_nodes;
            obj.activation_function = activation_function;
            obj.weights = weights;
            obj.biases = biases;
        end

        function set_weights(obj, weights)
            obj.weights = weights;
        end

        function set_biases(obj, biases)
            obj.biases = biases;
        end

        function w = get_weights(obj)
            w = obj.weights;
        end

        function b = get_biases(obj)
            b = obj.biases;
        end

        function fn = get_activation_function(obj)
            fn = obj.activation_function;
        end

        function [a, d] = forward(obj, x)
            a = zeros(obj.num_nodes, 1);

            for i = 1:obj.num_nodes
                a(i) = dot(x, obj.weights(i, :)) + obj.biases(i);
            end

            % Apply the activation function if it is not empty
            if isempty(obj.activation_function)
                d = a;
            else
                d = obj.activation_function(a);
            end
        end
    end
end