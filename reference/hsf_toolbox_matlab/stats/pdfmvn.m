function pdf = pdfmvn(x,Mu,Sigma)
% Returns the pdf of the multivariate normal distribution N(Mu,Sigma)

if (cols(x) == 1) && (rows(x) == rows(Mu))
    x = x';
end

pdf = mvnpdf(x,Mu',Sigma);

end
