function [I_X,I_Y,I_XY,I_X_Y] = shannon_entropy(p_XY)
    if cols(p_XY) == 1
        I_X = -sumc(missrv(p_XY .* log(p_XY),0));
        I_Y = 0;
        I_XY = 0;
    else
        I_XY = -sumc(sumc(missrv(p_XY .* log(p_XY),0)));
        p_X = sumr(p_XY);
        p_Y = sumc(p_XY);
        I_X = -sumc(missrv(p_X .* log(p_X),0));
        I_Y = -sumc(missrv(p_Y .* log(p_Y),0));
    end

    I_X_Y = I_X + I_Y - I_XY;
end
