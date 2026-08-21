function [Lambda1,Lambda2] = estimate_markov_generator(Lambda)

    Lambda1 = diagrv(max(Lambda,0),diag(Lambda) + ...
            sumc(diagrv(min(Lambda,0),0)'));

    G = abs(diag(Lambda)) + sumc(diagrv(max(Lambda,0),0)');
    B = sumc(diagrv(max(-Lambda,0),0)');

    K = rows(Lambda);
    Lambda2 = Lambda;

    for i = 1:K
        for j = 1:K
            if i ~= j && Lambda(i,j) < 0
                Lambda2(i,j) = 0.0;
            elseif G(i) > 0
                Lambda2(i,j) = Lambda(i,j) - B(i)*abs(Lambda(i,j))/G(i);
            end
        end
    end
end
