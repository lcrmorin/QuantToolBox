function E_a = endemics_area_relationship(n_i,s_j,A,a)

if isempty(s_j)
    E_a = sum((a./A).^n_i,1);
else
    j = seqa(1,1,rows(s_j));
    E_a = sum(s_j .* (a./A).^j,1);
end

end
