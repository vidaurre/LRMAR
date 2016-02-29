function Z = lrmarinstant (X,T,model,standardize)

if nargin<4, standardize = 1; end

if any(abs(mean(X))>1e-5)
   warning('Data is being centered, consider standardizing as well')
   X = X - repmat(mean(X),size(X,1),1);
end
[~,Y] = formautoregr(X,T,model.train.P,model.train.L);

VPsi = model.V.Mu_V .* ...
    repmat(model.Psi.Gam_shape ./ model.Psi.Gam_rate,size(model.V.Mu_V,1),1); 
Prec = model.Omega.Gam_shape ./ model.Omega.Gam_rate;
Z = (Y * VPsi') / (diag(Prec) + VPsi * model.V.Mu_V');

if standardize, Z = Z ./ repmat(std(Z),size(Z,1),1); end

end