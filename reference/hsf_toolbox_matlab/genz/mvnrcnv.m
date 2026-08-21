function [ p e ] = mvnrcnv( m, rho, a, b ), n = length(a); p = 0; e = 0;
%
%  [ p e ] = mvnrcnv( m, r, a, b )
%    uses a randomized quasi-random rule with m points to estimate an
%    MVN probability for constant rho (>0) correlatiion matrix,
%    with lower integration limits a and upper integration limits b.
%   Probability p is output with error estimate e.
%    Example:
%     rho = .25; a = -inf*[1 1 1 1 ]'; b = [ 1 2 3 4 ]';
%     [ p e ] = mvnrcnv( 5000, rho, a, b ); disp([ p e ])

%
%   Reference: Y.L. Tong, The Multivariate Normal Distribution, pp. 192-3.
%
ns = 12; nv = round( max(m/ns,1) ); on = ones(1,nv);
rs = sqrt(rho); ros = sqrt(1-rho); bon = b*on/ros; aon = a*on/ros;
%
% Randomization loop for ns samples
%
for k = 1 : ns, x = ones(n,1)*([1:nv]/nv + rand*on);
  z = rs*Phinv( abs( 2*mod(x,1) - 1 ) )/ros; % periodizing transformation
  pk = mean( exp( sum( log( ( Phi(bon+z) - Phi(aon+z) ) ) ) ) );
  d = ( pk - p )/k; p = p + d;
  if abs(d) > 0, e = abs(d)*sqrt( 1 + (e/d)^2*(k-2)/k );
  else, if k > 1, e = e*sqrt((k-2)/k); end
  end
end, e = 3*e; % error estimate is 3 x standard error with ns samples
%
%  Standard statistical normal distribution functions
%
function p = Phi(z),   p =  erfc( -z/sqrt(2) )/2;
function z = Phinv(p), z = norminv( p );
% function z = Phinv(p), z = -sqrt(2)*erfcinv( 2*p ); % use if no norminv
%
