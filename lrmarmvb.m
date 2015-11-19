function model = lrmarmvb (X,model,Z)
%
% Variational Bayes M step 
%
% X - T x ndim data matrix
% model - LRMAR model
% Z - latent variables
%
% Author: Diego Vidaurre, OHBA, University of Oxford

[T,ndim]=size(X);

Q = model.train.Q;
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

GramX = XX' * XX;

%%% Input - Gamma: W
if 1
    sigmad = model.sigma.Gam_shape ./  model.sigma.Gam_rate';
    for j=1:Q
        model.W.S_W(j,:,:) = inv(diag(sigmad(:)) + (model.Omega.Gam_shape / model.Omega.Gam_rate(j)) * GramX);
        model.W.Mu_W(:,j) =  (model.Omega.Gam_shape / model.Omega.Gam_rate(j)) * permute(model.W.S_W(j,:,:),[2 3 1]) * XX' * Z.Mu_Z(:,j);
    end
end

%%% Input - hidden: Omega
%     model.Omega.Gam_rate = model.prior.Omega.Gam_rate;
%     zz = repmat(diag(Z.S_Z)', T-P-L+1, 1);
%     for k=1:K
%         m = permute(XW(k,:,:),[2 3 1]);
%         e = (Z.Mu_Z - m).^2;
%         swx2 = zeros(T-P,Q);
%         for j=1:Q
%             tmp = XX * permute(model.state(k).W.S_W(n,:,:),[2 3 1]);
%             swx2(:,j) = sum(tmp .* XX,2);
%         end;
%         model.Omega.Gam_rate = model.Omega.Gam_rate +  0.5* sum( repmat(Gamma(:,k),1,Q) .* (e + swx2 + zz) );
%     end;
%     model.Omega.Gam_shape = model.prior.Omega.Gam_shape + (T-P) / 2;

%%% Input - hidden: sigma
if 1
    model.sigma.Gam_shape = model.prior.sigma.Gam_shape + Q/2;
    for n=1:ndim,
        for i=1:P
            index = (i-1)*ndim+n;
            model.sigma.Gam_rate(i,n) = model.prior.sigma.Gam_rate(i,n) + ...
                0.5 * (model.W.Mu_W(index,:) * model.W.Mu_W(index,:)' + ...
                sum(model.W.S_W(:,index,index)));
        end;
    end;
end;

%%% Hidden - Output: V
SZZ = (T-P-L+1)*Z.S_Z;
EZZ = Z.Mu_Z' * Z.Mu_Z + SZZ;
if 1
    for n=1:(ndim*L),
        model.V.S_V(n,:,:) = inv(diag(model.gamma.Gam_shape ./ model.gamma.Gam_rate) + (model.Psi.Gam_shape / model.Psi.Gam_rate(n)) * EZZ);
        model.V.Mu_V(:,n) = (model.Psi.Gam_shape / model.Psi.Gam_rate(n)) * permute(model.V.S_V(n,:,:),[2 3 1]) * Z.Mu_Z' * Y(:,n);
    end;
end;

% Hidden - Output: Psi
if 1
    m = Z.Mu_Z * model.V.Mu_V;
    e = sum((Y - m).^2 );
    svz2 = zeros(1,ndim*L);
    for n=1:(ndim*L)
        svz2(n) = model.V.Mu_V(:,n)' * SZZ * model.V.Mu_V(:,n) + ...
            trace( permute(model.V.S_V(n,:,:),[2 3 1]) * EZZ);
    end;
    model.Psi.Gam_shape = model.prior.Psi.Gam_shape + (T-P-L+1) / 2;
    model.Psi.Gam_rate = model.prior.Psi.Gam_rate + 0.5*(e + svz2);
end;

%%% Hidden - Output: gamma
if 1
    model.gamma.Gam_shape = model.prior.gamma.Gam_shape + ndim/2;
    model.gamma.Gam_rate = zeros(1,Q);
    for j=1:Q,
        model.gamma.Gam_rate(j) = model.prior.gamma.Gam_rate(j) + ...
            0.5 * (model.V.Mu_V(j,:) * model.V.Mu_V(j,:)' + sum(model.V.S_V(:,j,j)));
    end;
end;

