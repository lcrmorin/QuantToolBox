function u = rndnCopula(r,c,mtd)
    if nargin == 1
        c = 1;
        mtd = 0;
    elseif nargin == 2
        mtd = 0;
    end
    if mtd == 0
        u = randn(r,c);
    else
        u = rand(r,c);
        u = cdfni(u);
    end
end
