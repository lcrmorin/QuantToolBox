function S = survivalExponential(t, lambda)
%SURVIVALEXPONENTIAL Survival function for the (piecewise) exponential model.
%
%   S = survivalExponential(t, lambda)
%
%   INPUTS
%     t      : vector of times, N x 1 (or 1 x N)
%     lambda : hazard specification. Two cases:
%
%                (a) Homogeneous exponential model:
%                    lambda is a vector of hazard rates (1 x C or C x 1),
%                    one per scenario/curve.
%                    -> S(t) = exp(-lambda * t)
%
%                (b) Piecewise-constant hazard model:
%                    lambda is an M x (1+C) matrix. Column 1 contains the
%                    knots t*_1 < t*_2 < ... < t*_M (the hazard is
%                    extended to +Inf beyond t*_{M-1}, i.e. the last
%                    interval is unbounded). Columns 2:end contain the
%                    piecewise hazard rates lambda_1, ..., lambda_M for
%                    each of the C scenarios/curves.
%
%   OUTPUT
%     S : N x C matrix of survival probabilities S(t) = Pr(tau > t)
%
%   Vectorized MATLAB port of the GAUSS procedure survivalExponential.

    t = t(:);

    if size(lambda, 2) == 1
        % --- Homogeneous exponential case ---------------------------
        lam = lambda(:)';              % 1 x C
        S   = exp(-t * lam);           % N x C
        return
    end

    % --- Piecewise-constant hazard case -----------------------------
    tm  = lambda(:, 1);                % M x 1 knots
    lam = lambda(:, 2:end);            % M x C hazard rates
    tm(end) = Inf;                     % last interval is unbounded

    tm0 = [0; tm(1:end-1)];            % left edge of each interval
    d   = tm - tm0;                    % interval durations (last = Inf)

    % Cumulative hazard accumulated *before* entering interval m
    Hcum = [zeros(1, size(lam, 2)); cumsum(lam(1:end-1, :) .* d(1:end-1, :), 1)];

    % Locate the interval containing each t: bins are [tm0(1),tm(1)), ...
    edges = [tm0; Inf];
    idx   = discretize(t, edges);
    idx(isnan(idx)) = 1;               % safeguard for t < 0

    H = Hcum(idx, :) + lam(idx, :) .* (t - tm0(idx));
    S = exp(-H);
end
