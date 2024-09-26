% load trained mlp model from file
% load the model
my_model = load('models\model_5_0.010_0.0010.mat', 'mlp');
x = -1:0.01:1;
y = 2 .* x .^ 2 + 1;
y_hat = my_model.mlp.forward(x);
figure("visible", "on");
plot(x, y, 'r', x, y_hat, 'b');
