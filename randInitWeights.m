function w = randInitWeights(l_in, l_out)

  epsilon = sqrt(6)/(sqrt(l_out + l_in));
  w = rand(l_out, 1 + l_in) * 2 * epsilon - epsilon;

end