function [XX,Y,GramX] = formautoregr(X,T,P,L)
%
% form regressor and response for the autoregression;
% residuals are assumed to have size T(in)-order in each trial 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford
 
N = length(T); ndim = size(X,2);
maxP = max(P);  

XX = []; Y = [];
for tr=1:N
    t0 = sum(T(1:tr-1)); 
    XX0 = zeros(T(tr)-maxP-L+1,length(P)*ndim);
    for i=1:length(P)
        o = P(i);
        XX0(:,(1:ndim) + (i-1)*ndim) = X(t0+maxP-o+1:t0+T(tr)-o,:);
    end
    Y0 = zeros(T(tr)-maxP-L+1,ndim*L);
    for i=1:L
        Y0(:,(1:ndim) + (i-1)*ndim) = X(t0+maxP+i:t0+T(tr)+i-L,:);
    end
    XX = [XX; XX0];
    Y = [Y; Y0];
end

GramX = XX' * XX;

end
