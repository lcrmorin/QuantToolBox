function [j,s,brk] = species_abundance_distribution(n_i,brk)

if nargin == 1

    j = unique(n_i);
    J = rows(j);
    s = zeros(J,1);
    for iter = 1:J
        s(iter) = sum(n_i == j(iter));
    end

elseif isa(brk,'double') == 0
    if lower(string(brk)) ~= "octave"
        return
    end

    n_max = max(n_i);
    K = nextpow2(n_max);
    k = seqa(1,1,K);

    brk = 2.^k - 1;
    s = zeros(K,1);
    for iter = 1:K
        if iter == 1
            cnd = n_i <= brk(iter);
        else
            cnd = (n_i > brk(iter-1)) & (n_i <= brk(iter));
        end
        s(iter) = sum(cnd);
    end
    j = k;

else

    J = rows(brk);
    j = zeros(J,1);
    s = zeros(J,1);
    for iter = 1:J
        if iter == 1
            cnd = n_i <= brk(iter);
            j(iter) = 0.5*(1 + brk(iter));
        else
            cnd = (n_i > brk(iter-1)) & (n_i <= brk(iter));
            j(iter) = 0.5*(brk(iter-1)+ brk(iter));
        end
        s(iter) = sum(cnd);
    end

end

end
