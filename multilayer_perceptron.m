classdef multilayer_perceptron
    properties
        layers
        learning_gain
        momentum_gain
    end

    methods
        function obj = multilayer_perceptron(layers, learning_gain, momentum_gain)
            arguments
                layers (:, 1) {mustBeA(layers, 'mlp_layer')}
                learning_gain (1, 1) {mustBeNumeric, mustBeInRange(learning_gain, 0, 1)}
                momentum_gain (1, 1) {mustBeNumeric, mustBeInRange(momentum_gain, 0, 1)}
            end

            obj.layers = layers;
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
            y = x;

            for i = 1:length(obj.layers)
                [a, y] = obj.layers(i).forward(y);
            end
        end

        function train(obj, x, y, epochs)
            arguments
                obj
                x (:, :) {mustBeNumeric}
                y (:, :) {mustBeNumeric}
                epochs (1, 1) {mustBeNumeric, mustBeGreaterThan(epochs, 0)}
            end
            
            for i = 1:epochs
                for j = 1:length(x)
                    % Forward pass collect activations and decisions from each layer
                    activations = cell(1, length(obj.layers));
                    decisions = cell(1, length(obj.layers));
                    d = x(j, :);

                    for k = 1:length(obj.layers)
                        [a, d] = obj.layers(k).forward(d);
                        activations{k} = a;
                        decisions{k} = d;
                    end

                    % Backward pass
                    output_error = 0.5 .* (y(j, :) - d) .^ 2;

                    for k = length(obj.layers):-1:1
                        decision_error = 
                    end
                end
            end
        end
    end
end