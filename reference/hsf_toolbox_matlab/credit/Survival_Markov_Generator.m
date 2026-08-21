function S = Survival_Markov_Generator(t, Lambda)
    K = size(Lambda, 2);
    n = length(t);
    S = zeros(n, K);

    for i = 1:n
        if t(i) == 0
            S(i, :) = ones(1, K);
        else
            M = expm(t(i) * Lambda);
            S(i, :) = 1 - M(:,K)';
        end
    end
end
