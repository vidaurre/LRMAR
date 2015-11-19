function model = lrmarmvb (XX,Y,model,Z,GramX,update_omega)
%
% Variational Bayes M step
%
% XX and Y - autoregressors matrix and response
% model - LRMAR model
% Z - latent variables
% GramX = XX' * XX
%
% Author: Diego Vidaurre, OHBA, University of Oxford

[T,ndim]=size(Y);
Q = model.train.Q;
P = model.train.P;
L = model.train.L;
ndim = ndim / L;

%%% Input - Gamma: W
R = model.sigma.Gam_shape ./  model.sigma.Gam_rate';
R = diag(R(:));
for j=1:Q
    prec = model.Omega.Gam_shape / model.Omega.Gam_rate(j);
    model.W.S_W(j,:,:) = inv(R + prec * GramX);
    model.W.Mu_W(:,j) =  prec * permute(model.W.S_W(j,:,:),[2 3 1]) * XX' * Z.Mu_Z(:,j);
end

%%% Input - hidden: Omega   
if update_omega
    model.Omega.Gam_rate = model.prior.Omega.Gam_rate;
    e = sum((Z.Mu_Z - XX * model.W.Mu_W).^2);
    s = zeros(1,Q);
    for j=1:Q
        s(j) = T * Z.S_Z(j,j) + ...
            sum(sum( permute(model.W.S_W(j,:,:),[2 3 1]) .* GramX) );
    end;
    model.Omega.Gam_rate = model.Omega.Gam_rate + 0.5 * (e + s);
    model.Omega.Gam_shape = model.prior.Omega.Gam_shape + 0.5 * T;
end

%%% Input - hidden: sigma
model.sigma.Gam_shape = model.prior.sigma.Gam_shape + 0.5 * Q;
for n=1:ndim,
    for i=1:length(P)
        index = (i-1)*ndim+n;
        model.sigma.Gam_rate(i,n) = model.prior.sigma.Gam_rate(i,n) + ...
            0.5 * (sum(model.W.Mu_W(index,:).^2 + sum(model.W.S_W(:,index,index))));
    end
end

%%% Hidden - Output: V
SZZ = T*Z.S_Z;
EZZ = Z.Mu_Z' * Z.Mu_Z + SZZ;
R = diag(model.gamma.Gam_shape ./ model.gamma.Gam_rate);
for n=1:(ndim*L)
    prec = model.Psi.Gam_shape / model.Psi.Gam_rate(n);
    model.V.S_V(n,:,:) = inv( R + prec * EZZ);
    model.V.Mu_V(:,n) = prec * permute(model.V.S_V(n,:,:),[2 3 1]) * Z.Mu_Z' * Y(:,n);
end

%%% Hidden - Output: Psi
m = Z.Mu_Z * model.V.Mu_V;
e = sum((Y - m).^2 );
s = zeros(1,ndim*L);
for n=1:(ndim*L)
    s(n) = model.V.Mu_V(:,n)' * SZZ * model.V.Mu_V(:,n) + ...
        sum(sum( permute(model.V.S_V(n,:,:),[2 3 1]) .* EZZ));
end
model.Psi.Gam_shape = model.prior.Psi.Gam_shape + 0.5 * T;
model.Psi.Gam_rate = model.prior.Psi.Gam_rate + 0.5 * (e + s);

%%% Hidden - Output: gamma
model.gamma.Gam_shape = model.prior.gamma.Gam_shape + 0.5 * ndim;
model.gamma.Gam_rate = zeros(1,Q);
for j=1:Q,
    model.gamma.Gam_rate(j) = model.prior.gamma.Gam_rate(j) + ...
        0.5 * (sum(model.V.Mu_V(j,:).^2) + sum(model.V.S_V(:,j,j)));
end
