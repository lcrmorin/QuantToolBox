function [CB1,CB2] = carbon_budget_Reduction(t0,t,CE_t0,R,mtd)

if mtd == 3
    % Growth rate
    CB1 = (1-exp(-R*(t-t0)))/R*CE_t0;
    fun = @(s) CE_t0*exp(-R * (s-t0));
    CB2 = integral(fun,t0,t);
elseif mtd == 2
    % Compound rate
    CB1 = ((1-R)^(t-t0) - 1)/(log(1-R))*CE_t0;
    fun = @(s) CE_t0*((1 - R).^(s-t0));
    CB2 = integral(fun,t0,t);
else
    % Linear rate
    CB1 = (t-t0)*CE_t0 - 0.5*((t-t0)^2)*R;
    fun = @(s) CE_t0 - R*(s-t0);
    CB2 = integral(fun,t0,t);
end

end
