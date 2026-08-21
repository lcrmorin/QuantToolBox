function ytm = compute_bond_ytm(t,Ct,P)
    a = 0;
    b = 1;
    while (b-a) > 1e-5
        c = (a+b)/2;
        Pc = compute_bond_price(t,Ct,c,1);

        if Pc > P
            a = c;
        else
            b = c;
        end
    end

    ytm = (a+b)/2;
end
