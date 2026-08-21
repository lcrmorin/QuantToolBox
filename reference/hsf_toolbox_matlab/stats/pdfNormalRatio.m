function [p,a_z,b_z,c,rho_z] = pdfNormalRatio(z,mu_x,sigma_x, mu_y, sigma_y)
    mu2_x = mu_x.^2;
    mu2_y = mu_y.^2;
    sigma2_x = sigma_x.^2;
    sigma2_y = sigma_y.^2;
    z2 = z.^2;

    a_z = sqrt(z2./sigma2_x + 1/sigma2_y);
    b_z = (mu_x./sigma2_x).*z + (mu_y./sigma2_y);
    c = mu2_x./sigma2_x + mu2_y./sigma2_y;
    rho_z = z./(sigma_x.*a_z);

    a2_z = a_z.^2;
    a3_z = a_z.^3;
    b2_z = b_z.^2;

    p1 = b_z./(sigma_x.*sigma_y.*sqrt(2*pi).*a3_z);
    p2 = cdfn(b_z./a_z) - cdfn(-b_z./a_z);
    p3 = exp((b2_z - c.*a2_z)./(2*a2_z));
    p4 = exp(-c/2)./(sigma_x.*sigma_y.*a2_z.*pi);
    p = p1 .* p2 .* p3 + p4;
return;
