function [qf,Q,R,c,results] = quadratic_form_bond_portfolio1(sector,varphi_MD,MD,MD_star,...
    varphi_DTS,DTS,DTS_star,gamma_carry,carry,w)

    [qf_MD,Q_MD,R_MD,c_MD,results_MD] = quadratic_form_risk(sector,MD,MD_star,w);
    [qf_DTS,Q_DTS,R_DTS,c_DTS,results_DTS] = quadratic_form_risk(sector,DTS,DTS_star,w);

    Q = varphi_MD * Q_MD + varphi_DTS * Q_DTS;
    R = gamma_carry * carry + varphi_MD * R_MD + varphi_DTS * R_DTS;
    c = varphi_MD * c_MD + varphi_DTS * c_DTS;
    qf = quadratic_form(w,Q,R,c);

    results.MD = results_MD;
    results.DTS = results_DTS;
    results.Q = Q;
    results.R = R;
    results.c = c;
    results.qf = qf;
end
