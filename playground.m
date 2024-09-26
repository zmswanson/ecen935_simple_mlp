epochs = 1:1:1000;
mse = rand(1, length(epochs));
n_hidden = 5;
learning_gain = 20;
momentum_gain = 20;

test_x = -1:0.01:1;
test_y = (2 .* (test_x .^ 2)) + 1;

y_hat = rand(1, length(test_x));

figure("visible", "off")
semilogy(epochs, mse);
xlabel('Epochs');
ylabel('MSE');
ttl = sprintf('MSE over epochs\nn = %d, learning gain = %.3f, momentum gain = %.3f', n_hidden, learning_gain, momentum_gain);
title(ttl);
filename = sprintf('figures/mse_%d_%.3f_%.3f.png', n_hidden, learning_gain, momentum_gain);
% save without the popup window
saveas(gcf, filename)

figure("visible", "off")
plot(test_x, test_y, 'r', test_x, y_hat, 'b');
xlabel('x');
ylabel('y');
new_title = sprintf('Test-data vs Predictions\nn = %d, learning gain = %.3f, momentum gain = %.3f', n_hidden, learning_gain, momentum_gain);
title(new_title);
filename = sprintf('figures/function_fitting_%d_%f_%f.png', n_hidden, learning_gain, momentum_gain);
saveas(gcf, filename);