function [p, pm] = predict(X, Theta1, Theta2)

  a1 = [ones(size(X)(1), 1), X];

  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2)(1), 1), a2];

  z3 = a2 * Theta2';
  pm = sigmoid(z3);

  [v p] = max(pm, [], 2);
end