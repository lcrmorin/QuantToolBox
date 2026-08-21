function [qf,Q_b,R_b,c_b,results] = quadratic_form_bond_portfolio2(sector,varphi_AS,varphi_MD,MD,MD_star,...
    varphi_DTS,DTS,DTS_star,gamma_carry,carry,w,b)

    unique_sector = unique(sector);
    nSector = rows(unique_sector);
    if isempty(MD_star)
        MD_star = zeros(nSector,1);
    end
    if isempty(DTS_star)
        DTS_star = zeros(nSector,1);
    end

    [qf_MD,Q_MD,R_MD,c_MD,results_MD] = quadratic_form_risk(sector,MD,MD_star,w);
    [qf_DTS,Q_DTS,R_DTS,c_DTS,results_DTS] = quadratic_form_risk(sector,DTS,DTS_star,w);

    n = rows(w);
    Q_AS_b = eye(n);
    R_AS_b = b;
    c_AS_b = 0.5 * (b'*b);

    Q_MD_b = Q_MD;
    R_MD_b = R_MD + Q_MD * b;
    c_MD_b = 0.5 * b' * Q_MD * b + b'*R_MD + c_MD;

    Q_DTS_b = Q_DTS;
    R_DTS_b = R_DTS + Q_DTS * b;
    c_DTS_b = 0.5 * b' * Q_DTS * b + b'*R_DTS + c_DTS;

    Q_b = varphi_AS * Q_AS_b + varphi_MD * Q_MD_b + varphi_DTS * Q_DTS_b;
    R_b = gamma_carry * carry + varphi_AS * R_AS_b + varphi_MD * R_MD_b + varphi_DTS * R_DTS_b;
    c_b = gamma_carry*b'*carry + varphi_AS * c_AS_b +  varphi_MD * c_MD_b + varphi_DTS * c_DTS_b;
    qf = quadratic_form(w,Q_b,R_b,c_b);

    results.MD = results_MD;
    results.DTS = results_DTS;

    results.Q = Q_b;
    results.R = R_b;
    results.c = c_b;
    results.qf = qf;

    results.Q_AS_b = Q_AS_b;
    results.R_AS_b = R_AS_b;
    results.c_AS_b = c_AS_b;

    results.Q_MD_b = Q_MD_b;
    results.R_MD_b = R_MD_b;
    results.c_MD_b = c_MD_b;

    results.Q_DTS_b = Q_DTS_b;
    results.R_DTS_b = R_DTS_b;
    results.c_DTS_b = c_DTS_b;

end
