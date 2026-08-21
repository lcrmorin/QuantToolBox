function [qf,Q,R,c,results] = quadratic_form_risk(sector,Risk,Risk_star,w)

    n = rows(Risk);

    unique_sector = unique(sector);
    nSector = rows(unique_sector);
    results.n = n;
    results.nSector = nSector;
    results.unique_sector = unique_sector;

    Q = 0;
    R = 0;
    c = 0;
    results.Q_j = zeros(n,n,nSector);
    results.R_j = zeros(n,nSector);
    results.c_j = zeros(nSector,1);
    for j = 1:nSector
        s_j = double(sector == unique_sector(j));
        s_j_Risk = s_j .* Risk;

        Q_j = s_j_Risk * s_j_Risk';
        R_j = s_j_Risk * Risk_star(j);
        c_j = 0.5*(Risk_star(j).^2);

        Q = Q + Q_j;
        R = R + R_j;
        c = c + c_j;

        results.Q_j(:,:,j) = Q_j;
        results.R_j(:,j) = R_j;
        results.c_j(j) = c_j;
    end

    results.Q = Q;
    results.R = R;
    results.c = c;

    if nargin == 4
        qf = quadratic_form(w,Q,R,c);
    else
        qf = NaN;
    end

    results.qf = qf;

end
