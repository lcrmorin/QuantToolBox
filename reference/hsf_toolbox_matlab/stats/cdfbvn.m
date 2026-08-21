function p = cdfbvn(x,y,rho)

    [r,c,e] = max_size(x,y,rho);
    x = -x .* e;
    y = -y .* e;
    rho = rho .* e;

    p = zeros(r,c);

    for i = 1:r
        for j = 1:c
            p(i,j) = bvnu(x(i,j),y(i,j),rho(i,j));
        end
    end

end
