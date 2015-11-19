function [model,Z,fehist]=lrmartrain(X,options)
%
% Train LRMAR model using using the Variational Framework
%
% INPUTS:
%
% X - T x ndim data matrix
% options.Q - Number of latent components
% options.P - Order the MAR model
% options.L - Output order (no. of outputs lags to consider)
% options.cyc - maximum number of cycles of VB inference (default 100)
% options.tol - termination tol (change in log-evidence) (default 0.0001)
% options.showfe - show free energy progress? 
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
if ~isfield(options,'showfe'), options.showfe = 1; end

%%%% Init model
[model,Z] = lrmarinit(X,options);

fehist=[];
fe=0;
for cycle=1:model.train.cyc
    
    %%%% Compute free energy
    oldfe=fe;
    fe=evalfreeenergy(X,model,Z);        
    fehist=[fehist; fe];
    mesgstr='';
    if cycle>2
        if (fe-oldfe) > 0,
            mesgstr='(Violation)';
        end;
        if abs((fe - oldfe)/oldfe*100) < model.train.tol
            break;
        end;
    end;
    if options.showfe
        fprintf('cycle %i free energy = %f %s \n',cycle,fe,mesgstr);
    end
    
    %%%% VB E-step update
    Z = lrmarevb(X,model);
    
    %%%% VB M-step update
    model=lrmarmvb(X,model,Z);
end

return;
