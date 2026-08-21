function p = cdfig(x,mu,lambda)
    alpha = (x - mu)./ mu .* sqrt(lambda ./ x);
    alpha_bar = -(x + mu)./ mu .* sqrt(lambda ./ x);
    p = cdfn(alpha) + exp(2*lambda./mu) .* cdfn(alpha_bar);
end
