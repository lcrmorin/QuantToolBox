function [CB1,CB2,CB3] = carbon_budget_piecewise(t0,t,t_k,CE_k)

fun = @(t) interp1(t_k,CE_k,t,'linear');
CE_t0 = fun(t0);
CE_t = fun(t);

cnd = (t_k >= t0) & (t_k <= t);
t_k = t_k(cnd);
CE_k = CE_k(cnd);

if t_k(1) ~= t0
    t_k = [t0; t_k];
    CE_k = [CE_t0; CE_k];
end

if t_k(end) ~= t
    t_k = [t_k; t];
    CE_k = [CE_k; CE_t];
end

t_k1 = t_k(1:end-1);
t_k2 = t_k(2:end);
dt_k = t_k2 - t_k1;

CE_k1 = CE_k(1:end-1);
CE_k2 = CE_k(2:end);

beta0 = (t_k2./dt_k) .* CE_k1 - (t_k1./dt_k) .* CE_k2;
beta1 = (CE_k2 - CE_k1)./dt_k;

CB1 = beta0 .* (t_k2-t_k1) + (1/2)*beta1.*((t_k2.^2)-(t_k1.^2));
CB1 = sum(CB1);

if nargout > 1
    CB2 = integral(fun,t0,t);
end

CB3 = sum(CE_k1 .* t_k2 - CE_k2 .* t_k1) + 0.5*sum((CE_k2 - CE_k1).*(t_k1 + t_k2));

end
