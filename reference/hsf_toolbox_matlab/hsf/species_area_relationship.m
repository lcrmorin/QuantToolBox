function S_a = species_area_relationship(n_i,s_j,A,a)

if isempty(s_j)
    S = size(n_i,1);
    S_a = S - sum((1 - a./A).^n_i,1);
else
    S = sum(s_j,1);
    j = seqa(1,1,rows(s_j));
    S_a = S - sum(s_j .* (1 - a./A).^j,1);
end

end
