function [year,data,carbon_emissions,carbon_target,carbon_nze,slopes] = cdp_filter(universe,estimate_trend,rescale_trend)

datFileName = 'data/chap9_cdp3';
S = load(datFileName,'year');
year = S.year;
S = load(datFileName,'ID');
ID = S.ID;
S = load(datFileName,'all_data');
all_data = S.all_data;
S = load(datFileName,'Region1','Region2','Region3');
Region1 = S.Region1;
Region2 = S.Region2;
Region3 = S.Region3;
S = load(datFileName,'IEA_Sector','GICS_Sector');
IEA_Sector = S.IEA_Sector;
%GICS_Sector = S.GICS_Sector;

n = rows(ID);

if isnumeric(universe)
    issuers = universe;
    indx = indnv(issuers,ID);
    if isnan(indx)
        warning('error: bad specification of issuers');
        return
    end
else
    universe = lower(universe);
    if strcmp(universe,'global')
        cnd = true(n,1);
    elseif strcmp(universe,'dm')
        cnd = strcmp(Region2,'DM');
    elseif strcmp(universe,'em')
        cnd = strcmp(Region2,'EM');
    elseif strcmp(universe,'emu')
        cnd = strcmp(Region3,'EMU');
    elseif strcmp(universe,'nam')
        cnd = strcmp(Region3,'North America');
    elseif strcmp(universe,'eu-ex-emu')
        cnd = strcmp(Region3,'Europe-ex-EMU');
    elseif strcmp(universe,'other-dm')
        cnd = strcmp(Region3,'Other DM');
    elseif strcmp(universe,'asia')
        cnd = strcmp(Region1,'Asia');
    elseif strcmp(universe,'europe')
        cnd = strcmp(Region1,'Europe');
    elseif strcmp(universe,'pacific')
        cnd = strcmp(Region1,'Pacific');
    elseif strcmp(universe,'region-other')
        cnd = strcmp(Region1,'South America') | strcmp(Region1,'Other');
    elseif strcmp(universe,'electricity')
        cnd = strcmp(IEA_Sector,'Power');
    elseif strcmp(universe,'industry')
        cnd = strcmp(IEA_Sector,'Industry');
    elseif strcmp(universe,'transport')
        cnd = strcmp(IEA_Sector,'Transport');
    elseif strcmp(universe,'iea-other')
        cnd = strcmp(IEA_Sector,'Other');
    end
    indx = seqa(1,1,n);
    indx = indx(cnd);
end

nYear = rows(year);
n = rows(indx);
data = zeros(nYear,6,n);

for i = 1:n
    y = all_data(:,:,indx(i));
    y = y ./ y(1,:);

    y = [y(:,1) y];
    cnd = year > 2020;
    y(cnd,1) = NaN;
    cnd = year < 2020;
    y(cnd,2) = NaN;
    cnd = year < 2019;
    y(cnd,3) = NaN;
    cnd = year < 2018;
    y(cnd,4) = NaN;
    data(:,:,i) = y;
end

% CE, Trend 2020, Trend 2019, Trend 2018, Targets, NZE

slopes = NaN(n,3);

if estimate_trend == 1
    for i = 1:n
        trends = NaN(nYear,3);

        x = year;
        cnd = year <= 2020;
        x = x(cnd);
        y = data(cnd,1,i);
        x_2020 = x(end);
        x_2019 = x(end-1);
        x_2018 = x(end-2);
        y_2020 = y(end);
        y_2019 = y(end-1);
        y_2018 = y(end-2);

        if sum(isnan(y)) > 2
            data(:,2:4,i) = trends;
            continue;
        end
        [x,y] = packr(x,y);
        x = [ones(rows(x),1) x];
        for iter = 3:-1:1
            if iter == 3
                x_last = x_2020;
                y_last = y_2020;
            elseif iter == 2
                x_last = x_2019;
                y_last = y_2019;
            elseif iter == 1
                x_last = x_2018;
                y_last = y_2018;
            end
            beta = inv(x'*x)*x'*y;
            if rescale_trend == 1
                beta(1) = y_last - beta(2)*x_last;
            end
            y_hat = beta(1) + beta(2)*year;
            if rescale_trend == 2
                y_hat(rows(x)) = y_last;
            end
            trends(:,iter) = y_hat;
            slopes(i,iter) = beta(2);
            x = x(1:end-1,:);
            y = y(1:end-1);
        end

        cnd = year < 2020;
        trends(cnd,3) = NaN;
        cnd = year < 2019;
        trends(cnd,2) = NaN;
        cnd = year < 2018;
        trends(cnd,1) = NaN;

        data(:,2:4,i) = trends(:,[3; 2; 1]);
    end
end

slopes = slopes(:,[3; 2; 1]);

carbon_emissions = reshape(data(:,1,:),nYear,n);
carbon_target = reshape(data(:,5,:),nYear,n);
carbon_nze = reshape(data(:,6,:),nYear,n);

end
