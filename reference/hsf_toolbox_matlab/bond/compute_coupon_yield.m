function yield = compute_coupon_yield(t,Ct,Rt)
    P = compute_bond_price(t,Ct,Rt,1);
    yield = Ct(1)/P;
end
