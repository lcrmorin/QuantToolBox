function varrho = SpearmanCopula(C)
  varrho = 12 * integral2(C,0,1,0,1) - 3;
end
