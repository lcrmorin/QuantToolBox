function f = pdfExponential(t, lambda)
%PDFEXPONENTIAL Density function for the (piecewise) exponential model.
%
%   f = pdfExponential(t, lambda)
%
%   See survivalExponential.m for the definition of t and lambda.
%
%   OUTPUT
%     f : N x C matrix of density values f(t) = lambda_m(t) * S(t)
%
%   NOTE: the original GAUSS pdfExponential mixed lambda[i] (linear
%   index) and lambda[i,.] (row index) inside the accumulation loop,
%   which silently produced wrong results whenever lambda had more than
%   one scenario column. The vectorized approach below is immune to
%   this class of bug because lam(idx,:) always selects full rows.

    S = survivalExponential(t, lambda);

    if size(lambda, 2) == 1
        lam = lambda(:)';
        f = lam .* S;
        return
    end

    t   = t(:);
    tm  = lambda(:, 1);
    lam = lambda(:, 2:end);
    tm(end) = Inf;

    tm0   = [0; tm(1:end-1)];
    edges = [tm0; Inf];
    idx   = discretize(t, edges);
    idx(isnan(idx)) = 1;

    f = lam(idx, :) .* S;
end
