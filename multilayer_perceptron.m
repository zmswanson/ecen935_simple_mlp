%{
    File: multilayer_perceptron.m
    Author: Zachary M Swanson
    Date: 09-25-2024
    Description: This class defines a multilayer perceptron (MLP) with a specified number of input,
                 hidden, and output nodes, as well as learning and momentum gains. The class provides
                 methods to set the learning and momentum gains, perform the forward pass through the
                 MLP, train the MLP, and evaluate the performance.
    Usage: Create an instance of the class with the desired parameters, e.g.,
                mlp = multilayer_perceptron(n_input, n_hidden, n_output, learning_gain, momentum_gain);
            Perform the forward pass through the MLP using the forward method, e.g.,
                y = mlp.forward(x);
            Train the MLP using the train method, e.g.,
                train_mse = mlp.train(train_x, train_y);
            Evaluate the performance of the MLP using the evaluate method, e.g.,
                mse = mlp.evaluate(test_x, test_y);
    Notes: The MLP uses the zms_sigmoid activation function defined in activation_funcs.m and the
           mlp_layer class defined in mlp_layer.m.

           This a very simple implementation of a multilayer perceptron (MLP) for educational
           and much more work would be necessary to extend it to more complex architectures and
           more diverse activation functions.
%}

classdef multilayer_perceptron < handle
    properties
        hidden_layer
        output_layer
        learning_gain
        momentum_gain
    end

    methods
        function obj = multilayer_perceptron(n_input, n_hidden, n_output, learning_gain, momentum_gain)
            arguments
                n_input (1, 1) {mustBeNumeric, mustBeGreaterThan(n_input, 0)}
                n_hidden (1, 1) {mustBeNumeric, mustBeGreaterThan(n_hidden, 0)}
                n_output (1, 1) {mustBeNumeric, mustBeGreaterThan(n_output, 0)}
                learning_gain (1, 1) {mustBeNumeric, mustBeInRange(learning_gain, 0, 1)}
                momentum_gain (1, 1) {mustBeNumeric, mustBeInRange(momentum_gain, 0, 1)}
            end

            rng(42); % Set random seed for reproducibility
            obj.hidden_layer = mlp_layer(n_hidden, @activation_funcs.zms_sigmoid, rand(n_hidden, n_input)); %, ones(n_hidden, 1));
            obj.output_layer = mlp_layer(n_output, [], rand(n_output, n_hidden)); %, ones(n_output, 1));

            obj.learning_gain = learning_gain;
            obj.momentum_gain = momentum_gain;
        end

        function set_learning_gain(obj, learning_gain)
            obj.learning_gain = learning_gain;
        end

        function set_momentum_gain(obj, momentum_gain)
            obj.momentum_gain = momentum_gain;
        end

        function y = forward(obj, x)
            y = zeros(length(x), 1);
            for i = 1:length(x)
                [dummy_a, d] = obj.hidden_layer.forward(x(i));
                [dummy_a, y(i)] = obj.output_layer.forward(d);
            end
        end

        function train_mse = train(obj, x, y)
            arguments
                obj
                x (:, :) {mustBeNumeric}
                y (:, :) {mustBeNumeric}
            end

            prev_hidden_delta = zeros(size(obj.hidden_layer.weights));
            prev_output_delta = zeros(size(obj.output_layer.weights));

            crnt_hidden_delta = zeros(size(obj.hidden_layer.weights));
            crnt_output_delta = zeros(size(obj.output_layer.weights));

            train_mse = 0;
            
            for i = 1:length(x)
                % Forward pass collect activations and decisions from each layer
                [a, d] = obj.hidden_layer.forward(x(i));
                [dummy_a, y_hat] = obj.output_layer.forward(d);

                % Backward pass
                e_y = y(i) - y_hat;
                train_mse = train_mse + 0.5 .* e_y .^ 2;
                decision_error = -1 .* (obj.output_layer.weights' * e_y);
                
                % Append a 1 to the activations for the hidden layer to account for the bias
                d = [d; 1];
                activation_error = d .* (1 - d) .* decision_error;
                activation_error = activation_error(1:end-1); % Remove the bias error

                x_i = [x(i); 1];
                
                crnt_hidden_delta = obj.learning_gain .* (-1 .* (activation_error * x_i')) + obj.momentum_gain .* prev_hidden_delta;
                crnt_output_delta = obj.learning_gain .* (e_y * d') + obj.momentum_gain .* prev_output_delta;

                % Update weights (biases are not updated)
                obj.hidden_layer.weights = obj.hidden_layer.weights + crnt_hidden_delta;
                obj.output_layer.weights = obj.output_layer.weights + crnt_output_delta;

                prev_hidden_delta = crnt_hidden_delta;
                prev_output_delta = crnt_output_delta;
            end

            train_mse = train_mse / length(x);
        end
        
        function mse = evaluate(obj, x, y)
            arguments
                obj
                x (:, :) {mustBeNumeric}
                y (:, :) {mustBeNumeric}
            end

            mse = 0;

            for i = 1:length(x)
                y_hat = obj.forward(x(i));
                mse = mse + 0.5 .* (y(i) - y_hat) .^ 2;
            end

            mse = mse / length(x);
        end
    end
end