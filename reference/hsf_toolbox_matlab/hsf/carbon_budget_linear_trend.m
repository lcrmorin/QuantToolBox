function [CB1,CB2] = carbon_budget_linear_trend(t0,t,beta0,beta1)

fun = @(s) beta0 + beta1 * s;
CB1 = integral(fun,t0,t);
CB2 = (1/2)*beta1*(t^2-t0^2)+beta0*(t-t0);

end
