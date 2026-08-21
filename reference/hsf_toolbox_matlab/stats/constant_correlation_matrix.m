function C = constant_correlation_matrix(n,rho)
    C = diagrv(rho*ones(n,n),1);
end
