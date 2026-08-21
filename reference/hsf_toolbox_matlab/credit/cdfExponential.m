function F = cdfExponential(t, lambda)
%CDFEXPONENTIAL Cumulative distribution function for the (piecewise)
%exponential model. F(t) = 1 - S(t).
%
%   F = cdfExponential(t, lambda)
%
%   See survivalExponential.m for the definition of t and lambda.

    F = 1 - survivalExponential(t, lambda);
end
