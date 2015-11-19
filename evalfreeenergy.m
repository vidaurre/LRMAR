function FrEn = evalfreeenergy (XX,Y,model,Z)
%
% Computes the Free Energy of an LRMAR model
%
% INPUT
%
% XX and Y - autoregressors matrix and response
% model - LRMAR model
% Z - Latent signal
%
% Author: Diego Vidaurre, OHBA, University of Oxford

[T,ndim]=size(Y);
Q = model.train.Q;
P = model.train.P;
L = model.train.L;
ndim = ndim / L;

% Neg-Entropy of Z
Entr = 0.5*T*Q*(1+log(2*pi)) + 0.5*T*logdet(Z.S_Z,'chol');

% KL-divergences
sigmaKL = 0;
for n=1:ndim
    for i=1:length(P)
        sigmaKL = sigmaKL + gamma_kl(model.sigma.Gam_shape,model.prior.sigma.Gam_shape, ...
            model.sigma.Gam_rate(i,n),model.prior.sigma.Gam_rate(i,n));
    end
end;
WKL = 0;
sigmad = model.sigma.Gam_shape ./  model.sigma.Gam_rate'; sigmad = sigmad(:);
for j=1:Q
    WKL = WKL + gauss_kl(model.W.Mu_W(:,j),zeros(ndim*length(P),1), permute(model.W.S_W(j,:,:),[2 3 1]), diag(sigmad));
end;
gammaKL = 0;
for j=1:Q
    gammaKL = gammaKL + gamma_kl(model.gamma.Gam_shape, model.prior.gamma.Gam_shape,...
        model.gamma.Gam_rate(j),model.prior.gamma.Gam_rate(j));
end;
VKL = 0;
for n=1:(ndim*L)
    VKL = VKL + gauss_kl(model.V.Mu_V(:,n),zeros(Q,1),permute(model.V.S_V(n,:,:),[2 3 1]), ...
        diag(model.gamma.Gam_rate ./ model.gamma.Gam_shape));
end;
OmegaKL = 0;
for j=1:Q
   OmegaKL = OmegaKL + gamma_kl(model.Omega.Gam_shape, model.prior.Omega.Gam_shape,...
       model.Omega.Gam_rate(j),model.prior.Omega.Gam_rate(j));
end;
PsiKL = 0;
for n=1:ndim
    PsiKL = PsiKL + gamma_kl(model.Psi.Gam_shape,model.prior.Psi.Gam_shape, ...
        model.Psi.Gam_rate(n),model.prior.Psi.Gam_rate(n));
end;
KLdiv = WKL + sigmaKL + VKL + gammaKL + OmegaKL + PsiKL;

% average log-likelihood for Y
ltpi1= ndim*L/2 * log(2*pi);
PsiWish_alphasum1=(ndim*L)*0.5*psi(model.Psi.Gam_shape);
ldetWishB1=0;
for n=1:(ndim*L),
    ldetWishB1=ldetWishB1+0.5*log(model.Psi.Gam_rate(n));
end;
mean = Z.Mu_Z * model.V.Mu_V; % + repmat(model.Mean.Mu,T-P,1)
d = (Y - mean);
Cd = repmat((model.Psi.Gam_shape ./ model.Psi.Gam_rate)',1,T) .* d';   
dist1=zeros(T,1);
for n=1:ndim,
    dist1=dist1-0.5*d(:,n).*Cd(n,:)';
end
NormWishtrace1=zeros(T,1);
for n=1:(ndim*L),
    NormWishtrace1 = NormWishtrace1 + 0.5 * (model.Psi.Gam_shape / model.Psi.Gam_rate(n)) * ...
        ( sum( (Z.Mu_Z * permute(model.V.S_V(n,:,:),[2 3 1])) .* Z.Mu_Z, 2) + ...
        model.V.Mu_V(:,n)' * Z.S_Z * model.V.Mu_V(:,n) + ...
        sum(sum( permute(model.V.S_V(n,:,:),[2 3 1]) .* (Z.S_Z) )) );
end;
avLLY = T * (-ltpi1-ldetWishB1+PsiWish_alphasum1) + sum(dist1 - NormWishtrace1);

% average log-likelihood for Z
ltpi2= Q/2 * log(2*pi);
% PsiWish_alphasum2 = 0; % Not modelled - set to identity
% ldetWishB2 = 0;
PsiWish_alphasum2=Q*0.5*psi(model.Omega.Gam_shape);
ldetWishB2=0;
for j=1:Q,
   ldetWishB2=ldetWishB2+0.5*log(model.Omega.Gam_rate(j));
end;
mean = XX * model.W.Mu_W;
d = ( Z.Mu_Z - mean);
Cd = repmat((model.Omega.Gam_shape ./ model.Omega.Gam_rate)',1,T) .* d';   
dist2=zeros(T,1);
for j=1:Q,
    dist2=dist2-0.5*d(:,j).*Cd(j,:)';
end
NormWishtrace2=zeros(T,1);
for j=1:Q,
    NormWishtrace2 = NormWishtrace2 + 0.5 * (model.Omega.Gam_shape / model.Omega.Gam_rate(j)) * ...
        ( sum( (XX * permute(model.W.S_W(j,:,:),[2 3 1])) .* XX,2) + Z.S_Z(j,j));
end;
avLLZ = T * (-ltpi2-ldetWishB2+PsiWish_alphasum2) + sum(dist2 - NormWishtrace2);

FrEn= -Entr + KLdiv - avLLZ - avLLY;
%[-Entr +KLdiv -avLLZ -avLLY]
