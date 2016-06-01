%
% Neural network model for iris dataset
% Author: Evgeniy Kuznetsov
%

% Load dataset
load ./irisData.mat;


% Split dataset
data = irisData;
data = data(:, [2, 3, 4, 5, 6]);
all_X = data(:, 1:(size(data)(2) - 1));
all_y = data(:, size(data)(2));
classes = unique(all_y);

[train_X, test_X, cv_X, train_y, cv_y, test_y] = splitData(data, classes);


% Learn
num_features = size(all_X)(2);
num_classes = length(classes);
l1_size = l2_size = num_features;
l3_size = num_classes;

Theta1 = randInitWeights(l1_size, l2_size);
Theta2 = randInitWeights(l2_size, l3_size);

init_params = [Theta1(:); Theta2(:)];

opts = optimset('MaxIter', 200);

costFunction = @(p) costFunction(p, train_X, train_y, ...
  num_classes, l1_size, l2_size, l3_size);

[thetas, cost] = fmincg(costFunction, init_params, opts);

Theta1 = reshape(thetas(1:(l2_size * (l1_size + 1))), l2_size, (l1_size + 1));
Theta2 = reshape(thetas((l2_size * (l1_size + 1) + 1):end), l3_size, (l2_size + 1));

fprintf("Learned!.\n");
fprintf('Program paused. Press any key to continue.\n');
pause;

% Validate


% Predict
disp("Predictions.");

for i = 1:length(test_X)
  [prediction pm] = predict(test_X(i, :), Theta1, Theta2);
  actual = test_y(i);
  fprintf("Prediction/actual %i = %i.\n", prediction, actual);
endfor





