function [Call_BS, Put_BS] = Black_Scholes_Model(S0,K,sigma,T,b,r)

    d1 = (log(S0 ./ K) + (b + 0.5*sigma.^2).*T) ./ (sigma.*sqrt(T));
    d2 = d1 - sigma.*sqrt(T);

    Call_BS = S0 .* exp((b-r).*T).*normcdf(d1) - K .* exp(-r.*T) .* normcdf(d2);
    Put_BS = -S0 .* exp((b-r).*T).*normcdf(-d1) + K .* exp(-r.*T) .* normcdf(-d2);
end
