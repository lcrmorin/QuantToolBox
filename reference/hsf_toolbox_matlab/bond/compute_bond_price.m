function P = compute_bond_price(t,Ct,Rt,mtd)

    if nargin == 3
        mtd = 1;
    end

    if mtd == 1
        Bt = exp(-t .* Rt);
    else
        Bt = 1 ./ ((1 + Rt)^t);
    end

    P = sum(Ct .* Bt);
end
