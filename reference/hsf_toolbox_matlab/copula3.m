%% Loi log-normale

clear;
clc;
close all;

init_global;

% Parametres des distributions
mu = [0.781; 115.8; 2];
sigma = [0.220; 175; 108];
p = size(mu,1);

% Correlation entre les variables aléatoires
rho = [1.00  0.85  0.00;
       0.85  1.00 -0.50;
       0.00 -0.50  1.00];
Sigma = rho;
mu = zeros(p,1);

% Simulation des observations
n = 100000;
P = chol(Sigma);
g = mu' + randn(n,p) * P;
u = normcdf(g);
X = exp(mu' + u .* sigma');

disp('Observations');
disp(X(1:10,:));
disp('-------------------------------');

% Matrice de correlation empirique
rho_hat1 = corr(X);
disp('Matrice de correlation empirique');
disp(rho_hat1);
disp('-------------------------------');

% Matrice de correlation de Kendall/Spearman
% tau = 2 * asin(rho) ./ pi;
% varrho = 6 * asin(rho/2) ./ pi;
% tau = corr(X,'type','Kendall');
% rho_hat2 =  sin(pi / 2 * tau);

varrho = corr(X,'type','Spearman');
rho_hat2 = 2 * sin(pi / 6 * varrho);

disp('Matrice de correlation (estimateur de Spearman)');
disp(rho_hat2);
disp('-------------------------------');

% Maximum de vraisemblance
mu_hat = mean(log(X));
sigma_hat = std(log(X));
U = logncdf(X,mu_hat .* ones(n,p),sigma_hat .* ones(n,p));
G = norminv(U);
rho_hat3 = corr(G);

disp('Estimateur du maximum de vraisemblance');
disp(rho_hat3);
disp('-------------------------------');
