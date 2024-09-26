rng(42); % Set random seed for reproducibility

n_input = 1;
n_hidden = 3;
n_output = 1;

hidden_layer = mlp_layer(n_hidden, @activation_funcs.zms_sigmoid, rand(n_hidden, n_input), ones(n_hidden, 1));
[dummy_a, dummy_d] = hidden_layer.forward(rand(n_input, 1));
disp(dummy_a);
disp(dummy_d);

% output_layer = mlp_layer(n_output, [], rand(n_output, n_hidden), ones(n_output, 1));
% disp(hidden_layer.weights);
% disp(output_layer.weights);

% mlp = multilayer_perceptron([hidden_layer, output_layer], 0.1, 0.9);
% output = mlp.forward(rand(n_input, 1));
% disp(output);