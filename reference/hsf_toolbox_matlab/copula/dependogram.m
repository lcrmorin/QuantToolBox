function u = dependogram(data)

n = rows(data);
m = cols(data);
u = zeros(n,m);
for j = 1:m
    x = data(:,j);
    r = tiedrank(x);
    u(:,j) = r/(n+1);
end

end
