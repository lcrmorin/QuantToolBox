function sac = hurlbert(n_i,m,mtd)

nSpecies = size(n_i,1);
n = sum(n_i);
sac = 0;
for s = 1:nSpecies
    if (n-n_i(s)) >= m
        if mtd == 2
            q = gammaln(n-n_i(s)+1) + gammaln(n-m+1) - (gammaln(n+1) + gammaln(n-n_i(s)-m+1));
            q = exp(q);
        else
            q = nchoosek(n-n_i(s),m)/nchoosek(n,m);
        end
        sac = sac + (1 - q);
    else
        sac = sac + 1;
    end
end

end
