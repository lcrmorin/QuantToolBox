function [I_X,I_Y,I_XY,I_X_Y] = shannon_entropy_markov_chain(Lambda,t)

    p_XY = expm(Lambda*1000);
    p_X = p_XY(1,:)';
    p_Y = p_X;

    nt = rows(t);

    I_X = zeros(nt,1);
    I_Y = zeros(nt,1);
    I_XY = zeros(nt,1);

    for i = 1:nt
        if t(i) == 0
            p_XY = eye(rows(Lambda));
        else
            p_XY = expMatrix(Lambda*t(i));
        end
        I_X(i) = -sumc(missrv(p_X .* log(p_X),0));
        I_Y(i) = -sumc(missrv(p_Y .* log(p_Y),0));
        I_XY(i) = -sumc(p_X .* sumr(missrv(p_XY .* log(p_XY),0)));
    end

    I_X_Y = I_X + I_Y - I_XY;
 end
