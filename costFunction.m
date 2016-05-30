function [J grad] = costFunction(params, X, y, num_classes, l1_size, l2_size, l3_size)

  m = length(y);

  Theta1 = reshape(params(1:(l2_size * (l1_size + 1))), l2_size, l1_size + 1);
  Theta2 = reshape(params((l2_size * (l1_size + 1) + 1):end), l3_size, l2_size + 1);

  % y to martix
  Y = zeros(length(y), num_classes);
  for i = 1:length(y)
    Y(i, y(i)) = 1;
  endfor

  % Cost function
  [v hypo] = predict(X, Theta1, Theta2);

  J = ((-Y) .* log(hypo)) - ((1 - Y) .* log(hypo));
  J = sum(sum(J));
  J = 1/m * J;

  % Grad

end