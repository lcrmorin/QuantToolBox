function qf = quadratic_form(x,Q,R,c)
    qf = 0.5 * x'*Q*x - x'*R + c;
end
