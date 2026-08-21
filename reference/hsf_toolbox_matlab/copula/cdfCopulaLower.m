function C = cdfCopulaLower(u)
    C = max(sum(u,2)-cols(u)+1,0);
end
