%{
    File: homework1.m
    Author: Zachary M Swanson
    Date: 09-25-2024
    Description: This script trains a multilayer perceptron (MLP) to fit a quadratic function, as
                 described in homework 1 for ECEN 935 (Computational Intelligence) at UNL. The
                 script performs a train-test split, trains the MLP with various hyperparameters,
                 and evaluates the performance. The results are saved as plots and in a CSV file.
    Usage: Run this script in MATLAB to train the MLP and generate the results.
%}
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
% n_hidden = 5;
n_output = 1;
% learning_gain = 0.01; % [0.01, 1]
% momentum_gain = 0.001;  % [0, 1]

% learning_gain = [0.01, 0.1, 0.5, 0.7, 0.9,  1];
% momentum_gain = [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.7, 0.9, 1];

learning_gain = [0.01];
momentum_gain = [0.1];

for n_hidden = [15, 20, 25, 30, 50, 100]
    for i = learning_gain
        for j = momentum_gain
            fprintf('--------------------------------------------------------------------------------\n');
            fprintf('Training model with n_hidden = %d, learning_gain = %.3f, momentum_gain = %.4f\n', n_hidden, i, j);
            mlp = multilayer_perceptron(n_input, n_hidden, n_output, i, j);

            epochs = [1:10000];
            mse = zeros(1, length(epochs));

            for k = epochs
                mse(k) = mlp.train(train_x, train_y);
                if mod(k, 1000) == 0
                    fprintf('Epoch %d: MSE = %f\n', k, mse(k));
                end
            end

            min_mse = min(mse);
            avg_mse = mean(mse);
            test_mse = mlp.evaluate(test_x, test_y);

            % plot the MSE over the epochs on a log scale... write plot to file
            figure("visible", "off");
            semilogy(epochs, mse);
            xlabel('Epochs');
            ylabel('MSE');
            new_title = sprintf('MSE over epochs\nn = %d, learning gain = %.3f, momentum gain = %.4f', n_hidden, i, j);
            title(new_title);
            filename = sprintf('figures/mse_%d_%.3f_%.4f.png', n_hidden, i, j);
            saveas(gcf, filename);


            % plot the full dataset against the predictions
            y_hat = mlp.forward(test_x);
            figure("visible", "off");
            plot(test_x, test_y, 'r', test_x, y_hat, 'b');
            xlabel('x');
            ylabel('y');
            new_title = sprintf('Test-data vs Predictions\nn = %d, learning gain = %.3f, momentum gain = %.4f', n_hidden, i, j);
            title(new_title);
            filename = sprintf('figures/function_fitting_%d_%.3f_%.4f.png', n_hidden, i, j);
            saveas(gcf, filename);

            % append the min and avg MSE to the file
            filename = 'mse_results.csv';
            fileID = fopen(filename, 'a');
            fprintf(fileID, '%d,%.3f,%.4f,%.8f,%.8f,%.8f\n', n_hidden, i, j, min_mse, avg_mse, test_mse);
            fclose(fileID);

            % save hidden and output layer weights
            filename = sprintf('models/model_%d_%.3f_%.4f.mat', n_hidden, i, j);
            save(filename, 'mlp');
        end
    end
end