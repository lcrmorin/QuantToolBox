function [ p, e ] = qsilatmvtv( m, nu, sigma, a, b )
%
%[ p, e ] = qsilatmvtv( m, nu, sigma, a, b )
%    Qsimvtv estimates a multivariate t-probability using an
%        m point randomized quasi-random integration method
%    for a t-distribution with
%        nu degrees-of-freedom (nu <=0 for multivariate normal), and
%        sigma, a positive definite covariance matrix.
%    The probability is computed over a hyper-rectangle specified with
%        a, lower integration limits and
%        b, upper integration limits.
%      If sigma is nxn then a and b must be column n-vectors.
%    Output for qsimvtv is
%        p, the estimated probability, along with
%        e, an absolute error estimate for p.
%    Example:
%     r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5];
%     a = -inf(4,1); b = [ 1 2 3 4 ]';
%     [ p e ] = qsilatmvtv( 5000, 5, r, a, b ); disp([ p e ])

%
%   This function uses an algorithm given in the paper
%      "Comparison of Methods for the Numerical Computation of
%       Multivariate t Probabilities", in
%      J. of Computational and Graphical Stat., 11(2002), pp. 950-971, by
%          Alan Genz and Frank Bretz
%
%   The primary references for the numerical integration are
%    "Randomization of Number Theoretic Methods for Multiple Integration",
%     R. Cranley & T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
%   and
%    "Fast Component-by-Component Construction, a Reprise for Different
%     Kernels", D. Nuyens & R. Cools. In H. Niederreiter and D. Talay,
%     editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004,
%     Springer-Verlag, 2006, 371-385.
%
%   Alan Genz is the author of this function and following Matlab functions.
%          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
%          Email : alangenz@wsu.edu
%
% Copyright (C) 2013, Alan Genz,  All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided the following conditions are met:
%   1. Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
%   2. Redistributions in binary form must reproduce the above copyright
%      notice, this list of conditions and the following disclaimer in
%      the documentation and/or other materials provided with the
%      distribution.
%   3. The contributor name(s) may not be used to endorse or promote
%      products derived from this software without specific prior
%      written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
% FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
% COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
% INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
% OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
% TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% Initialization
%
sn = max([1 sqrt(nu)]); [ch,as,bs] = chlrps(sigma,a/sn,b/sn); N = 10;
[n,n] = size(sigma); [q,P] = fstrnk(m/N,n); q = q/P; % lattice rule gens.
on = ones(1,P); p = 0; e = 0;
%
% Randomization loop for ns samples
%
for S = 1 : N, vp = on; s = zeros(n,P);
  for i = 1 : n, x = abs(2*mod(q(i)*[1:P]+rand,1)-1); % periodizing transform
    if i == 1, r = on; if nu > 0, r = sqrt(2*gaminv(x,nu/2,1)); end
    else, y = Phinv(c+x.*dc); s(i:n,:) = s(i:n,:) + ch(i:n,i-1)*y;
    end, si = s(i,:); c = on; ai = as(i)*r - si; d = on; bi = bs(i)*r - si;
    c(find(ai<=-9)) = 0; tl = find(abs(ai)<9); c(tl) = Phi(ai(tl));
    d(find(bi<=-9)) = 0; tl = find(abs(bi)<9); d(tl) = Phi(bi(tl));
    dc = d - c; vp = vp.*dc;
  end, d = ( mean(vp) - p )/S; p = p + d; e = ( S - 2 )*e/S + d^2;
end, e = 3*sqrt(e); % error estimate is 3 times standard error with ns samples.
% end qsimvtv
%
%
%  Standard statistical normal distribution functions
%
function p =   Phi(z), p =  erfc( -z/sqrt(2) )/2;
function z = Phinv(p), z = norminv( p );
%
function [ c, ap, bp ] = chlrps( R, a, b )
%
%  Computes permuted and scaled lower Cholesky factor c for R which may be
%   singular, also permuting and scaling integration limit vectors a and b.
%
ep = 1e-10; % singularity tolerance;
%
[n,n] = size(R); c = R; ap = a; bp = b; d = sqrt(max(diag(c),0));
for i = 1 :  n
  if d(i) > 0, c(:,i) = c(:,i)/d(i); c(i,:) = c(i,:)/d(i);
    ap(i) = ap(i)/d(i); bp(i) = bp(i)/d(i);
  end
end, y = zeros(n,1); sqtp = sqrt(2*pi);
for k = 1 : n, im = k; ckk = 0; dem = 1; s = 0;
   for i = k : n
       if c(i,i) > eps, cii = sqrt( max( c(i,i), 0 ) );
          if i > 1, s = c(i,1:k-1)*y(1:k-1); end
          ai = (ap(i)-s)/cii; bi = (bp(i)-s)/cii; de = Phi(bi) - Phi(ai);
          if de <= dem, ckk = cii; dem = de; am = ai; bm = bi; im = i; end
       end
   end
   if im > k
     ap([im k]) = ap([k im]); bp([im k]) = bp([k im]); c(im,im) = c(k,k);
      t = c(im,1:k-1); c(im,1:k-1) = c(k,1:k-1); c(k,1:k-1) = t;
      t = c(im+1:n,im); c(im+1:n,im) = c(im+1:n,k); c(im+1:n,k) = t;
      t = c(k+1:im-1,k); c(k+1:im-1,k) = c(im,k+1:im-1)'; c(im,k+1:im-1) = t';
   end
   if ckk > ep*k, c(k,k) = ckk; c(k,k+1:n) = 0;
      for i = k+1 : n
         c(i,k) = c(i,k)/ckk; c(i,k+1:i) = c(i,k+1:i) - c(i,k)*c(k+1:i,k)';
      end
      if abs(dem) > ep
	y(k) = ( exp( -am^2/2 ) - exp( -bm^2/2 ) )/( sqtp*dem );
      else, y(k) = ( am + bm )/2;
	if am < -10, y(k) = bm; elseif bm > 10, y(k) = am; end
      end, c(k,1:k) = c(k,1:k)/ckk; ap(k) = ap(k)/ckk; bp(k) = bp(k)/ckk;
   else, c(k:n,k) = 0; y(k) = ( ap(k) + bp(k) )/2;
   end
end
%
% end chlrps
%
function [ z, n ] = fstrnk( ni, sm, om, gm, bt )
%
% Reference:
%   "Fast Component-by-Component Construction, a Reprise for Different
%     Kernels", D. Nuyens and R. Cools. In H. Niederreiter and D. Talay,
%     editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004,
%     Springer-Verlag, 2006, 371-385.
% Modifications to original by A. Genz, 05/07
% Typical Use:
%  om = @(x)x.^2-x+1/6; n = 99991; s = 100; gam = 0.9.^[1:s];
%  z = fastrank( n, s, om, gam, 1 + gam/3 ); disp([z'; e])
%
n = fix(ni); if ~isprime(n), pt = primes(n); n = pt(length(pt)); end
if nargin < 3, om = @(x)x.^2-x+1/6;
  bt = ones(1,sm); gm = [ 1 (4/5).^[0:sm-2] ];
end, q = 1; w = 1; z = [1:sm]'; m = ( n - 1 )/2; g = prmrot(n);
perm = [1:m]; for j = 1 : m-1, perm(j+1) = mod( g*perm(j), n ); end
perm = min( n - perm, perm ); c = om(perm/n); fc = fft(c);
for s = 2 : sm, q = q.*( bt(s-1) + gm(s-1)*c([w:-1:1 m:-1:w+1]) );
  [ es w ] = min( real( ifft( fc.*fft(q) ) ) ); z(s) = perm(w);
end
%
% end fstrnk
%
function r = prmrot(pin)
%
% find primitive root for prime p, might fail for large primes (p > 32e7)
%
p = pin; if ~isprime(p), pt = primes(p); p = pt(length(pt)); end
pm = p - 1; fp = unique(factor(pm)); n = length(fp); r = 2; k = 1;
while k <= n; d = pm/fp(k); rd = r;
  for i = 2 : d, rd = mod( rd*r, p ); end % computes r^d mod p
  if rd == 1, r = r + 1; k = 0; end, k = k + 1;
end
%
% prmrot
%
