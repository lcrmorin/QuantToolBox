function [CB1,CB2] = carbon_budget_linear(t0,t,beta0,beta1)

CB1 = beta0 * (t-t0) + (1/2)*beta1*(t^2-t0^2);

if nargout == 2
    fun = @(s) beta0 + beta1 * s;
    CB2 = integral(fun,t0,t);
end

end
