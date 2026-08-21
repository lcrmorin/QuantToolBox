function C = cdfCopulaGumbel(u1,u2,theta)
    C = exp(-((-log(u1)).^theta+(-log(u2)).^theta).^(1./theta));
end
