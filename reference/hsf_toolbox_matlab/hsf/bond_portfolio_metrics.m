function [MD_w,DTS_w,MD_j,DTS_j] = bond_portfolio_metrics(sector,MD,DTS,w)

    unique_sector = unique(sector);
    nSector = rows(unique_sector);

    MD_w = sum(w .* MD);
    DTS_w = sum(w .* DTS);

    MD_j = zeros(nSector,1);
    DTS_j = zeros(nSector,1);
    for j = 1:nSector
        s_j = double(sector == unique_sector(j));
        MD_j(j) = sum(s_j .* w .* MD);
        DTS_j(j) = sum(s_j .* w .* DTS);
    end

end
