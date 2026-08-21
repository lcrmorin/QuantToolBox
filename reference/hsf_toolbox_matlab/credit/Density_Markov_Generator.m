function pdf = Density_Markov_Generator(t, Lambda)
    K = size(Lambda, 2);
    n = length(t);
    pdf = zeros(n, K);

    for i = 1:n
        if t(i) == 0
            row = Lambda(:,K);
        else
            M = Lambda * expm(t(i) * Lambda);
            row = M(:,K);
        end
        pdf(i, :) = row';
    end
end
