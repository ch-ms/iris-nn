function [J grad] = costFunction(params, X, y, num_classes, l1_size, l2_size, l3_size)

  m = length(y);

  Theta1 = reshape(params(1:(l2_size * (l1_size + 1))), l2_size, (l1_size + 1));
  Theta2 = reshape(params((l2_size * (l1_size + 1) + 1):end), l3_size, (l2_size + 1));

  % y to martix
  Y = zeros(length(y), num_classes);
  for i = 1:length(y)
    Y(i, y(i)) = 1;
  endfor

  % Cost function
  [v a1 a2 a3 z2 z3] = predict(X, Theta1, Theta2);

  J = ((-Y) .* log(a3)) - ((1 - Y) .* log(a3));
  J = sum(sum(J));
  J = 1/m * J;

  % Grad
  DELTA_1 = zeros(size(Theta1));
  DELTA_2 = zeros(size(Theta2));

  for i = [1:m]
    act1 = a1(i, :);
    act2 = a2(i, :);
    act3 = a3(i, :);

    sdelta_3 = (act3 - Y(i, :))';
    sdelta_2 = (Theta2' * sdelta_3) .* sigmoidGradient(z2(i, :)');
    sdelta_2 = sdelta_2(2:end);

    DELTA_1 += sdelta_2 * act1;
    DELTA_2 += sdelta_3 * act2;

  endfor

  Theta1_grad = 1/m .* DELTA_1;
  Theta2_grad = 1/m .* DELTA_2;

  grad = [Theta1_grad(:); Theta2_grad(:)];
end