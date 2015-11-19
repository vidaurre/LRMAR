function Z = lrmarevb (X,model)
%
% Variational Bayes E step 
%
% X - T x ndim data matrix
% model - LRMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

[T,ndim]=size(X);
if length(X)~=T,
    X=X';
    [T,ndim]=size(X);
end;

P = model.train.P;
L = model.train.L;

XX = zeros(T-P-L+1,P*ndim);
for i=1:P
    XX(:,(1:ndim) + (i-1)*ndim) = X(P-i+1:T-i-L+1,:);
end;
Y = zeros(T-P-L+1,ndim*L);
for i=1:L
    Y(:,(1:ndim) + (i-1)*ndim) = X(P+i:T+i-L,:);
end;

VPsi = model.V.Mu_V * diag(model.Psi.Gam_shape ./ model.Psi.Gam_rate); 
Z = struct('Mu_Z',[],'S_Z',[]);
Z.S_Z = inv(diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate) + VPsi * model.V.Mu_V'); 
Z.Mu_Z = (Z.S_Z * ( diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate) * (XX * model.W.Mu_W)' + VPsi * Y' ))';




