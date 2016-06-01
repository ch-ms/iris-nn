function [p, a1, a2, a3, z2, z3] = predict(X, Theta1, Theta2)

  a1 = [ones(size(X)(1), 1), X];

  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2)(1), 1), a2];
  z2 = [ones(size(z2)(1), 1), z2];

  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  [v p] = max(a3, [], 2);
end