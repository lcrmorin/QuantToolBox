function [ p, e ] = qsimvnauto( r, a, b, ep, m ), n = length(a); e = 1e-15;
%  [ p e ] = qsimvnauto( r, a, b, ep, m )
%    uses randomized quasi-random rules to estimate an
%    MVN probability for a positive definite covariance matrix r,
%     with integration limits column vectors a and b.
%   Probability p is output with error estimate e.
%    Defaulted inputs are absolute accuracy request ep [1e-3]
%      and work limit m [1e5].
%   Example:
%     r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5];
%     a = -inf*[1 1 1 1 ]'; b = [ 1 2 3 4 ]';
%     [ p e ] = qsimvnauto( r, a, b ); disp([ p e ])
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
if     n == 1, p = Phi(b) - Phi(a);
elseif n == 2, p = bvng( a, b, r );
else
  if nargin < 4, ep = 1e-3; end, if nargin < 5, m = 1e5; end
  mi = min(m,n*1000); p = 0; e = 1; ei = 0; mt = 0;
  while e > ep & mt < m, mi = round(1.41*mi);
    [ pi, ei, mi ] = qsimvnv( mi, r, a, b ); mt = mt + mi;
    wt = 1/( 1 + (ei/e)^2 ); p = p + wt*( pi - p ); e = sqrt(wt)*ei;
  end
end
% end qsimvnauto
function [ p, e, nt ] = qsimvnv( m, r, a, b )
%
%  [ P E ] = QSIMVNV( M, R, A, B )
%    uses a randomized quasi-random rule with m points to estimate an
%    MVN probability for positive definite covariance matrix r,
%     with lower integration limit column vector a and upper
%     integration limit column vector b.
%   Probability p is output with error estimate e.
%    Example:
%     r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5];
%     a = -inf*[1 1 1 1 ]'; b = [ 1 2 3 4 ]';
%     [ p e ] = qsimvnv( 5000, r, a, b ); disp([ p e ])
%
%   This function uses an algorithm given in the paper
%      "Numerical Computation of Multivariate Normal Probabilities", in
%      J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
%          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
%          Email : alangenz@wsu.edu
%  The primary references for the numerical integration are
%   "On a Number-Theoretical Integration Method"
%   H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11, and
%   "Randomization of Number Theoretic Methods for Multiple Integration"
%    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
%
%   Alan Genz is the author of this function and following Matlab functions.
%
% Initialization
%
[ch as bs] = chlrdr(r,a,b); ct = ch(1,1); ai = as(1); bi = bs(1);
if ai > -9*ct, if ai < 9*ct, c = Phi(ai/ct); else, c=1; end, else c=0; end
if bi > -9*ct, if bi < 9*ct, d = Phi(bi/ct); else, d=1; end, else d=0; end
[n, n] = size(r); ci = c; dci = d - ci; p = 0; e = 0;
ns = 10; nv = fix( max( [ m/ns 1 ] ) ); on = ones(1,nv); y = zeros(n-1,nv);
ps = sqrt(primes(5*n*log(n)/4)); q = [1/nv ps(1:n-2)]'; % Richtmyer generators
%
% Randomization loop for ns samples
%
for j = 1 : ns, c = ci*on; dc = dci*on; pv = dc;
  for i = 2 : n, x = abs( 2*mod( q(i-1)*[1:nv] + rand, 1 ) - 1 );
    y(i-1,:) = Phinv( c + x.*dc ); s = ch(i,1:i-1)*y(1:i-1,:);
    ct = ch(i,i); ai = as(i) - s; bi = bs(i) - s; c = on; d = c;
    c(find( ai < -9*ct )) = 0; d(find( bi < -9*ct )) = 0;
    tstl = find( abs(ai) < 9*ct ); c(tstl) = Phi( ai(tstl)/ct );
    tstl = find( abs(bi) < 9*ct ); d(tstl) = Phi( bi(tstl)/ct );
    dc = d - c; pv = pv.*dc;
  end, d = ( mean(pv) - p )/j; p = p + d; e = ( j - 2 )*e/j + d^2;
end, e = 3*sqrt(e); nt = ns*nv; % error est is 3 x std error with ns samples.
%
% end qsimvnv
%
%
%  Standard statistical normal distribution functions
%
function p =   Phi(z), p =  erfc( -z/sqrt(2) )/2;
function z = Phinv(p), z = norminv( p );
%function z = Phinv(p), z = -sqrt(2)*erfcinv( 2*p ); % use if no norminv
%
function [ c, ap, bp ] = chlrdr( R, a, b )
%
%  Computes permuted lower Cholesky factor c for R which may be singular,
%   also permuting integration limit vectors a and b.
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
    if c(i,i) > eps, cii = sqrt( max( [c(i,i) 0] ) );
      if i > 1, s = c(i,1:k-1)*y(1:k-1); end
      ai = ( ap(i)-s )/cii; bi = ( bp(i)-s )/cii; de = Phi(bi) - Phi(ai);
      if de <= dem, ckk = cii; dem = de; am = ai; bm = bi; im = i; end
    end
  end
  if im > k, c(im,im) = c(k,k);
    ap([im k]) = ap([k im]); bp([im k]) = bp([k im]);
    c([im;k],1:k-1) = c([k;im],1:k-1); c(im+1:n,[im k]) = c(im+1:n,[k im]);
    t = c(k+1:im-1,k); c(k+1:im-1,k) = c(im,k+1:im-1)'; c(im,k+1:im-1) = t';
  end, c(k,k+1:n) = 0;
  if ckk > ep*k, c(k,k) = ckk;
    for i = k+1 : n
      c(i,k) = c(i,k)/ckk; c(i,k+1:i) = c(i,k+1:i) - c(i,k)*c(k+1:i,k)';
    end
    if abs(dem) > ep, y(k) = ( exp(-am^2/2) - exp(-bm^2/2) )/(sqtp*dem);
    else, y(k) = 0;
      if     am >  -9 & bm <  9, y(k) = ( am + bm )/2;
      elseif am <= -9 & bm <  9, y(k) = bm;
      elseif am >  -9 & bm >= 9, y(k) = am;
      end
    end
  else, c(k:n,k) = 0; y(k) = 0;
  end
end
%
% end chlrdr
%
function p = bvng( a, b, sg )
%
%  bvng( a, b, sg )
%  A function for computing a bivariate normal probability for a
%   bivariate normal x, with a < x < b and covariance matrix sg.
%  Example:
%   p = bvng([-inf -2],[5 inf],[4 1;1 3]);
%
  cx = sqrt(sg(1,1)); cy = sqrt(sg(2,2)); r = sg(2,1)/(cx*cy);
  xl = a(1)/cx; xu = b(1)/cx; yl = a(2)/cy; yu = b(2)/cy;
  p = bvnu(xl,yl,r) - bvnu(xu,yl,r) - bvnu(xl,yu,r) + bvnu(xu,yu,r);
% end bvng
function p = bvnu( dh, dk, r )
%BVNU
%  A function for computing bivariate normal probabilities.
%  bvnu calculates the probability that x > dh and y > dk.
%    parameters
%      dh 1st lower integration limit
%      dk 2nd lower integration limit
%      r   correlation coefficient
%  Example: p = bvnu( -3, -1, .35 )
%  Note: to compute the probability that x < dh and y < dk,
%        use bvnu( -dh, -dk, r ).
%

%   Author
%       Alan Genz
%       Department of Mathematics
%       Washington State University
%       Pullman, Wa 99164-3113
%       Email : alangenz@wsu.edu
%
%    This function is based on the method described by
%        Drezner, Z and G.O. Wesolowsky, (1989),
%        On the computation of the bivariate normal inegral,
%        Journal of Statist. Comput. Simul. 35, pp. 101-107,
%    with major modifications for double precision, for |r| close to 1,
%    and for Matlab by Alan Genz. Minor bug modifications 7/98, 2/10.
%
  if dh == inf | dk == inf, p = 0;
  elseif dh == -inf, if dk == -inf, p = 1; else p = Phi(-dk); end
  elseif dk == -inf, p = Phi(-dh);
  else
    if abs(r) < 0.3, ng = 1; lg = 3;
      %       Gauss Legendre points and weights, n =  6
      w(1:3,1) = [0.1713244923791705 0.3607615730481384 0.4679139345726904]';
      x(1:3,1) = [0.9324695142031522 0.6612093864662647 0.2386191860831970]';
    elseif abs(r) < 0.75,  ng = 2; lg = 6;
      %       Gauss Legendre points and weights, n = 12
      w(1:3,2) = [.04717533638651177 0.1069393259953183 0.1600783285433464]';
      w(4:6,2) = [0.2031674267230659 0.2334925365383547 0.2491470458134029]';
      x(1:3,2) = [0.9815606342467191 0.9041172563704750 0.7699026741943050]';
      x(4:6,2) = [0.5873179542866171 0.3678314989981802 0.1252334085114692]';
    else, ng = 3; lg = 10;
      %       Gauss Legendre points and weights, n = 20
      w(1:3,3) = [.01761400713915212 .04060142980038694 .06267204833410906]';
      w(4:6,3) = [.08327674157670475 0.1019301198172404 0.1181945319615184]';
      w(7:9,3) = [0.1316886384491766 0.1420961093183821 0.1491729864726037]';
      w(10,3) = 0.1527533871307259;
      x(1:3,3) = [0.9931285991850949 0.9639719272779138 0.9122344282513259]';
      x(4:6,3) = [0.8391169718222188 0.7463319064601508 0.6360536807265150]';
      x(7:9,3) = [0.5108670019508271 0.3737060887154196 0.2277858511416451]';
      x(10,3) = 0.07652652113349733;
    end
    h = dh; k = dk; hk = h*k; bvn = 0;
    if abs(r) < 0.925, hs = ( h*h + k*k )/2; asr = asin(r);
      for i = 1 : lg
	sn = sin( asr*( 1 - x(i,ng) )/2 );
	bvn = bvn + w(i,ng)*exp( ( sn*hk - hs )/( 1 - sn*sn ) );
	sn = sin( asr*( 1 + x(i,ng) )/2 );
	bvn = bvn + w(i,ng)*exp( ( sn*hk - hs )/( 1 - sn*sn ) );
      end, bvn = bvn*asr/( 4*pi );
      bvn = bvn + Phi(-h)*Phi(-k);
    else, twopi = 2*pi; if r < 0, k = -k; hk = -hk; end
      if abs(r) < 1, as = (1-r)*(1+r); a = sqrt(as); bs = (h-k)^2;
	c = ( 4 - hk )/8 ; d = ( 12 - hk )/16; asr = -( bs/as + hk )/2;
	if asr > -100
	  bvn = a*exp(asr)*( 1 - c*(bs-as)*(1-d*bs/5)/3 + c*d*as*as/5 );
	end
	if hk > -100, b = sqrt(bs); sp = sqrt(twopi)*Phi(-b/a);
	  bvn = bvn - exp(-hk/2)*sp*b*( 1 - c*bs*( 1 - d*bs/5 )/3 );
	end, a = a/2;
	for i = 1 : lg
	  for is = -1 : 2 : 1, xs = ( a + a*is*x(i,ng) )^2;
	    rs = sqrt( 1 - xs ); asr = -( bs/xs + hk )/2;
	    if asr > -100, sp = ( 1 + c*xs*( 1 + d*xs ) );
	      ep = exp( -hk*xs/( 2*(1+rs)^2 ) )/rs;
	      bvn = bvn + a*w(i,ng)*exp(asr)*( ep - sp );
	    end
	  end
	end, bvn = -bvn/twopi;
      end
      if r > 0, bvn =  bvn + Phi( -max( h, k ) );
      elseif h >= k, bvn = -bvn;
      else, if h < 0, L = Phi(k)-Phi(h); else, L = Phi(-h)-Phi(-k); end
        bvn =  L - bvn;
      end
    end, p = max( 0, min( 1, bvn ) );
  end
%
%   end bvnu
%
