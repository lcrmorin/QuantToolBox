function x = cdfSTi(p, xi, omega, eta, nu, tol, maxit)

if nargin  == 5
    tol     = 1e-8;
    maxit   = 50;
end

r = max(rows(p),rows(eta));
c = max(cols(p),cols(eta));
e = ones(r, c);
p = vecr(p .* e);
eta = vecr(eta .* e);
nu = vecr(nu .* e);

x = cdfti(p, nu);
q = cdfST(x, 0, 1, eta, nu);
e_max = q > 0.90;
e_min = q < 0.10;
x = (eta >= 0.0) .* (0.02 .* e_min + (1-e_min) .* x) + ...
  (eta < 0.0) .* (-0.02 .* e_max + (1-e_max) .* x);
dx = 1;

iters = 0;
while (max(max(abs(dx))) > tol) && (iters < maxit)
    dx = (cdfST(x,0,1,eta,nu) - p) ./ pdfST(x,0,1,eta,nu);
    x = x - dx;
    iters = iters + 1;
end

x = missex(x, abs(p-cdfST(x, 0, 1, eta, nu)) >= 0.01);
x = reshape(x, r, c);

x = xi + omega .* x;
end
