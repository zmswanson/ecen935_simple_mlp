rng(42); % Set random seed for reproducibility

x = [-1:0.01:1];
y = (2 .* (x .^ 2)) + 1;

% create train-test split of 80-20 by randomly shuffling the indices
indices = randperm(length(x));
train_indices = indices(1:round(0.8 * length(x)));
test_indices = indices(round(0.8 * length(x)) + 1:end);

% reorder the indices
train_indices = sort(train_indices);
test_indices = sort(test_indices);

train_x = x(train_indices);
train_y = y(train_indices);

test_x = x(test_indices);
test_y = y(test_indices);

fprintf('Training data size: %d\n', length(train_x));
fprintf('Test data size: %d\n', length(test_x));

n_input = 1;
n_hidden = 5;
n_output = 1;
learning_gain = 0.01; % [0.01, 1]
momentum_gain = 0.001;  % [0, 1]

lg = [0.01, 0.1, 0.5, 0.7, 0.9,  1];
mg = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7, 0.9, 1];

mlp = multilayer_perceptron(n_input, n_hidden, n_output, learning_gain, momentum_gain);

epochs = [1:10000];
mse = zeros(1, length(epochs));

for i = epochs
    mse(i) = mlp.train(train_x, train_y);

    fprintf('Epoch %d: MSE = %f\n', i, mse(i));
end

min_mse = min(mse);
avg_mse = mean(mse);

% plot the MSE over the epochs on a log scale... write plot to file
figure("visible", "off");
semilogy(epochs, mse);
xlabel('Epochs');
ylabel('MSE');
new_title = sprintf('MSE over epochs\nn = %d, learning gain = %.3f, momentum gain = %.3f', n_hidden, learning_gain, momentum_gain);
title(new_title);
filename = sprintf('figures/mse_%d_%.3f_%.3f.png', n_hidden, learning_gain, momentum_gain);
saveas(gcf, filename);


% plot the full dataset against the predictions
y_hat = mlp.forward(test_x);
figure("visible", "off");
plot(test_x, test_y, 'r', test_x, y_hat, 'b');
xlabel('x');
ylabel('y');
new_title = sprintf('Test-data vs Predictions\nn = %d, learning gain = %.3f, momentum gain = %.3f', n_hidden, learning_gain, momentum_gain);
title(new_title);
filename = sprintf('figures/function_fitting_%d_%f_%f.png', n_hidden, learning_gain, momentum_gain);
saveas(gcf, filename);

