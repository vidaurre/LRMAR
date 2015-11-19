function [model,Z,fehist]=lrmartrain(X,T,options)
%
% Train LRMAR model using using the Variational Framework
%
% INPUTS:
%
% X - time points x ndim data matrix
% T - number of time points per trial
% options.Q - Number of latent components
% options.P - Order the MAR model
% options.L - Output order (no. of outputs lags to consider)
% options.cyc - maximum number of cycles of VB inference (default 100)
% options.tol - termination tol (change in log-evidence) (default 0.0001)
% options.verbose - show free energy progress?
%
% OUTPUTS
% model - LRMAR model
% Z - latent variables
% fehist - historic of the free energies across iterations
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if ~isfield(options,'L'), options.L = 1; end
if ~isfield(options,'cyc'), options.cyc = 1000; end
if ~isfield(options,'tol'), options.tol = .001; end
if ~isfield(options,'verbose'), options.verbose = 1; end

if any(abs(mean(X)>1e-10))
   warning('Data is being centered, consider standardizing as well')
   X = X - repmat(mean(X),size(X,1),1);
end

% Auxiliar variables
[XX,Y,GramX] = formautoregr(X,T,options.P,options.L);

% Init model
[model,Z] = lrmarinit(XX,Y,options);

for cycle=1:model.train.cyc
        
    % VB E-step update
    Z = lrmarevb(XX,Y,model);
   
    % VB M-step update
    model=lrmarmvb(XX,Y,model,Z,GramX,1); %cycle==1);
    
    % Compute free energy
    fe=evalfreeenergy(XX,Y,model,Z);
    if cycle>=2
        ch = (fe - oldfe)/abs(oldfe);
        fehist=[fehist; fe];
        if abs(ch) < model.train.tol
            break;
        end
        if options.verbose
            fprintf('cycle %i free energy = %g (relative change %g) \n',cycle,fe,ch);
        end
    else
        fehist = fe;
        if options.verbose
            fprintf('cycle %i free energy = %g \n',cycle,fe);
        end
    end
    oldfe=fe;
    
end

end