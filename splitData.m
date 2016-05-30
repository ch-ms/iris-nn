function [train_X, test_X, cv_X, train_y, cv_y, test_y] = splitData(data, classes)

  data = data(randperm(size(data)(1)), :);

  y = data(:, size(data)(2));

  ratio = .12;
  ratio_p_class = ratio / length(classes);

  test = zeros(0, size(data)(2));
  cv = zeros(0, size(data)(2));
  train = zeros(0, size(data)(2));

  for i = 1:length(classes)
    idx = find(y == classes(i));

    class_data = data(idx, :);
    total = size(data, 1);
    els = floor((total * ratio_p_class) / 2);

    test_idx = 1:els;
    cv_idx = els+1:els*2+1;
    train_idx = els*2+2:size(class_data)(1);

    test = [test; class_data(test_idx, :)];
    cv = [cv; class_data(cv_idx, :)];
    train = [train; class_data(train_idx, :)];

  endfor

  test_X = test(:, 1:size(data)(2) - 1);
  test_y = test(:, size(data)(2));

  cv_X = cv(:, 1:size(data)(2) - 1);
  cv_y = cv(:, size(data)(2));

  train_X = train(:, 1:size(data)(2) - 1);
  train_y = train(:, size(data)(2));

end