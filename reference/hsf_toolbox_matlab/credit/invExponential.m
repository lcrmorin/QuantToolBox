function t = invExponential(p, lambda)
%INVEXPONENTIAL Quantile function (inverse CDF) for the (piecewise)
%exponential model.
%
%   t = invExponential(p, lambda)
%
%   INPUTS
%     p      : matrix of probabilities in (0,1), N x 1 or N x C
%     lambda : hazard specification, see survivalExponential.m
%
%   OUTPUT
%     t : N x C matrix of quantiles such that Pr(tau <= t) = p
%
%   Vectorized MATLAB port of the GAUSS procedure invExponential.

    tolp = eps;
    Sp   = 1 - p;                          % target survival level
    bad  = (Sp >= 1 - tolp) | (Sp <= tolp);
    Sp(bad) = 0.5;                         % placeholder, avoids log(0)/log(1)

    if size(lambda, 2) == 1
        lam = lambda(:)';
        t = -log(Sp) ./ lam;
        t(bad) = NaN;
        return
    end

    tm  = lambda(:, 1);
    lam = lambda(:, 2:end);
    C   = size(lam, 2);

    Sm = survivalExponential(tm, lambda);   % M x C, survival at each knot
    Sm(end, :) = 0;                          % force terminal survival to 0
    Sm  = [ones(1, C); Sm];                  % (M+1) x C, prepend S(0) = 1
    tm0 = [0; tm];                            % (M+1) x 1

    if size(Sp, 2) == 1 && C > 1
        Sp = repmat(Sp, 1, C);
    end

    N = size(Sp, 1);
    t = zeros(N, C);
    for c = 1:C
        % Number of knots whose survival exceeds the target level gives
        % the index of the bracketing interval (Sm is non-increasing).
        idx = sum(Sm(:, c) > Sp(:, c)', 1)';
        idx = max(idx, 1);

        S0 = Sm(idx, c);
        t0 = tm0(idx);
        L0 = lam(idx, c);

        t(:, c) = t0 + (log(S0) - log(Sp(:, c))) ./ L0;
    end

    t(bad) = NaN;
end
