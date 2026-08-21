function tau = rndExponential(r, c, lambda)
%RNDEXPONENTIAL Random default-time generator for the (piecewise)
%exponential model.
%
%   tau = rndExponential(r, c, lambda)
%
%   INPUTS
%     r, c   : if c ~= 0, generates an r x c matrix of uniforms and
%              simulates default times from it.
%              if c == 0, r is interpreted as a pre-generated matrix of
%              uniforms (mirrors the GAUSS calling convention).
%     lambda : hazard specification, see survivalExponential.m
%
%   OUTPUT
%     tau : simulated default times

    if c ~= 0
        u = rand(r, c);
    else
        u = r;
    end
    tau = invExponential(u, lambda);
end
