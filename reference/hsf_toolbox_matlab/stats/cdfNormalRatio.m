function [p,a_z,b_z,c,rho_z] = cdfNormalRatio(z,mu_x,sigma_x, mu_y, sigma_y)
    mu2_x = mu_x.^2;
    mu2_y = mu_y.^2;
    sigma2_x = sigma_x.^2;
    sigma2_y = sigma_y.^2;
    z2 = z.^2;

    a_z = sqrt(z2./sigma2_x + 1/sigma2_y);
    b_z = (mu_x./sigma2_x).*z + (mu_y./sigma2_y);
    c = mu2_x./sigma2_x + mu2_y./sigma2_y;
    rho_z = z./(sigma_x.*a_z);

    x1 = (mu_x - mu_y.*z)./(sigma_x.*sigma_y.*a_z);
    y1 = -mu_y./sigma_y;
    x2 = -x1;
    y2 = -y1;

    p = cdfbvn(x1,y1,rho_z) + cdfbvn(x2,y2,rho_z);
return;
