function [model,Z] = lrmarinit (XX,Y,options)
%
% Initialise LRMAR model
%
% INPUTS:
%
% X - T x ndim data matrix
% options.Q - Number of latent components
% options.P - Order the MAR model
% options.L - Output P (no. of outputs lags to consider)
% options.cyc - maximum number of cycles of VB inference (default 100)
% options.tol - termination tol (change in log-evidence) (default 0.0001)
%
% Author: Diego Vidaurre, OHBA, University of Oxford

model=struct('train',struct());
model.train.Q = options.Q;
model.train.P = options.P;
model.train.L = options.L;
model.train.cyc = options.cyc; 
model.train.tol = options.tol; 

model=initpriors(Y,model);
[model,Z]=initpost(XX,Y,model);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model] = initpriors(Y,model)
% Init priors

ndim=size(Y,2);
Q = model.train.Q;
P = model.train.P;
L = model.train.L;

model.prior.sigma = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.sigma.Gam_shape = 0.001;
model.prior.sigma.Gam_rate = 0.001*ones(length(P),ndim);
model.prior.Omega = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.Omega.Gam_shape = Q + 0.1 - 1;
model.prior.Omega.Gam_rate = (Q + 0.1 - 1)*ones(1,Q);
model.prior.Psi = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.Psi.Gam_shape = L + ndim + 0.1 - 2;
model.prior.Psi.Gam_rate = (L + ndim + 0.1 - 2)*ones(1,ndim*L);
model.prior.gamma = struct('Gam_shape',[],'Gam_rate',[]);
model.prior.gamma.Gam_shape = 0.001;
model.prior.gamma.Gam_rate = 0.001*ones(1,Q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model,Z] = initpost(XX,Y,model)
% Init posteriors

[T,ndim]=size(Y);
Q = model.train.Q;
P = model.train.P;
L = model.train.L;
ndim = ndim / L;

% Z 
[~, scores] = pca(Y, 'NumComponents', Q );
scores = scores - repmat(mean(scores),T,1);
scores = scores ./ repmat(std(scores),T,1);
Z = struct('Mu_Z',[],'S_Z',[]);
Z.S_Z = zeros(Q,Q);
Z.Mu_Z  = scores;

% W
model.W = struct('Mu_W',[],'S_W',[]);
model.W.S_W = zeros(Q,ndim*length(P),ndim*length(P));
model.W.S_W(1,:,:) = inv(XX'  * XX + 0.001 * eye(size(XX,2)));
for j=2:Q, model.W.S_W(j,:,:) = model.W.S_W(1,:,:); end;
model.W.Mu_W = permute(model.W.S_W(1,:,:),[2 3 1]) * XX' * Z.Mu_Z;

% V
model.V = struct('Mu_V',[],'S_V',[]);
model.V.S_V(1,:,:) = inv(Z.Mu_Z' * Z.Mu_Z + 0.1 * eye(size(Z.Mu_Z,2)) );
for n=2:(ndim*L),
    model.V.S_V(n,:,:) = model.V.S_V(1,:,:);
end;
model.V.Mu_V = permute(model.V.S_V(1,:,:),[2 3 1]) * Z.Mu_Z' * Y;

% Omega 
model.Omega.Gam_shape = model.prior.Omega.Gam_shape + 0.5 * T;
e1 = sum((Z.Mu_Z - XX * model.W.Mu_W).^2);
model.Omega.Gam_rate = model.prior.Omega.Gam_rate + 0.5 * e1;
% model.Omega.Gam_shape = 1;
% model.Omega.Gam_rate = ones(1,Q);

% Psi
model.Psi.Gam_shape = model.prior.Psi.Gam_shape + 0.5 * T;
model.Psi.Gam_rate = model.prior.Psi.Gam_rate;
e2 = sum((Y - Z.Mu_Z * model.V.Mu_V).^2 );
model.Psi.Gam_rate = model.prior.Psi.Gam_rate + 0.5 * e2;

% sigma
model.sigma.Gam_shape = model.prior.sigma.Gam_shape + Q/2;
model.sigma.Gam_rate = model.prior.sigma.Gam_rate;
for n=1:ndim,
    for i=1:length(P)
        index = (i-1)*ndim+n;
        model.sigma.Gam_rate(i,n) = model.sigma.Gam_rate(i,n) + 0.5 * model.W.Mu_W(index,:) * model.W.Mu_W(index,:)';
    end
end

% gamma
model.gamma.Gam_shape = model.prior.gamma.Gam_shape + 0.5*ndim;
model.gamma.Gam_rate = zeros(1,Q);
for j=1:Q,
    model.gamma.Gam_rate(j) = model.prior.gamma.Gam_rate(j) + 0.5 * (model.V.Mu_V(j,:) * model.V.Mu_V(j,:)');
end



