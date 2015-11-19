function [B] = obslike (X,T,n,hmm)

% function [B] = obslike (X,T,n,hmm)
%
% Evaluate likelihood of data given observation model
% for hmm.obsmodel = 'Gauss','Poisson','Dirichlet' or 'LIKE'
%
% X          N by p data matrix
% T          length of series to learn
% n          block index (time series data can be split into many blocks)
% hmm        hmm data structure
%
% B          Likelihood of N data points

[T,ndim]=size(X);
if length(X)~=T,
    X=X';
    [T,ndim]=size(X);
end;
K=hmm.K;


B=zeros(T,K);
Q=hmm.Q;
order=hmm.order;
L = hmm.L;

XX = zeros(T-order-L+1,order*ndim);
for i=1:order
    XX(:,(1:ndim) + (i-1)*ndim) = X(order-i+1:T-i-L+1,:);
end;
Y = zeros(T-order-L+1,ndim*L);
for i=1:L
    Y(:,(1:ndim) + (i-1)*ndim) = X(order+i:T+i-L,:);
end;

ltpi1= ndim/2 * log(2*pi);
ltpi2= Q/2 * log(2*pi);

ldetWishB1=0;
for n=1:ndim,
    ldetWishB1=ldetWishB1-0.5*digamma(hmm.Psi.Gam_shape)+0.5*log(hmm.Psi.Gam_rate(n));
end;
ldetWishB2=0;
for j=1:Q,
    ldetWishB2=ldetWishB2-0.5*digamma(hmm.Omega.Gam_shape)+0.5*log(hmm.Omega.Gam_rate(j));
end;

for k=1:K
    hs=hmm.state(k);
    % likelihood for Y
    mean = hs.Z.Mu_Z * hmm.V.Mu_V; % + repmat(hs.Mean.Mu,T-order,1);
    d = (Y - mean);
    Cd = diag(hmm.Psi.Gam_shape ./ hmm.Psi.Gam_rate) * d';
    dist=zeros(T-order-L+1,1);
    for n=1:ndim,
        dist=dist-0.5*d(:,n).*Cd(n,:)';
    end
    NormWishtrace=zeros(T-order-L+1,1);
    for n=1:(ndim*L),
        NormWishtrace = NormWishtrace + 0.5 * (hmm.Psi.Gam_shape / hmm.Psi.Gam_rate(n)) * ...
            sum( (hs.Z.Mu_Z * permute(hmm.V.S_V(n,:,:),[2 3 1])) .* hs.Z.Mu_Z', 2) + ...
            0.5 * (hmm.Psi.Gam_shape / hmm.Psi.Gam_rate(n)) * hs.Z.S_Z;
    end;
    
    B(order+1:T-L+1,k)= -ldetWishB1-ltpi1 + dist - NormWishtrace;
    % likelihood for Z
    mean = XX * hs.W.Mu_W;
    d = ( hs.Z.Mu_Z - mean);
    Cd = diag(hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate) * d';
    dist=zeros(T-order-L+1,1);
    for j=1:Q,
        dist=dist-0.5*d(:,j).*Cd(j,:)';
    end
    NormWishtrace=zeros(T-order-L+1,1);
    for j=1:Q,
        NormWishtrace = NormWishtrace + 0.5 * (hmm.Omega.Gam_shape / hmm.Omega.Gam_rate(j)) * ...
            sum( (XX * permute(hs.W.S_W(j,:,:),[2 3 1])) .* XX', 2) + ...
            0.5 * (hmm.Omega.Gam_shape / hmm.Omega.Gam_rate(j)) * hs.Z.S_Z(j,j);
    end   
    B(order+1:T-L+1,k)= B(order+1:T-L+1,k) -ldetWishB2-ltpi2 + dist - NormWishtrace;
end

B=exp(B);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dist] = mdist (x,mu,C)
%
%   [dist] =  mdist(x,mu,C)
%
%   computes from x values given mean mu and precision C
%   the distance, actually the quantity
%
%        -0.5  (x-mu)' C (x-mu)

d=size(C,1);
if (size(x,1)~=d)  x=x'; end;
if (size(mu,1)~=d)  mu=mu'; end;

[ndim,N]=size(x);

d=x-mu*ones(1,N);
Cd=C*d;

% less expensive
dist=zeros(1,N);
for l=1:ndim,
    dist=dist+d(l,:).*Cd(l,:);
end
dist=-0.5*dist';

return;

