function varrho = SpearmanCopulaFrank(theta)
  varrho = 1 - 12 .* (DebyeFunction(theta,1)-DebyeFunction(theta,2)) ./ theta;
end
