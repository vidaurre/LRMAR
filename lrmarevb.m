function Z = lrmarevb (XX,Y,model)
%
% Variational Bayes E step 
%
% XX and Y - autoregressors matrix and response
% model - LRMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

VPsi = model.V.Mu_V .* ...
    repmat(model.Psi.Gam_shape ./ model.Psi.Gam_rate,size(model.V.Mu_V,1),1); 

Z = struct('Mu_Z',[],'S_Z',[]);
Prec = model.Omega.Gam_shape ./ model.Omega.Gam_rate;
Z.S_Z = inv(diag(Prec) + VPsi * model.V.Mu_V'); 
Z.Mu_Z = ( repmat(Prec,size(XX,1),1) .* (XX * model.W.Mu_W) + Y * VPsi') * Z.S_Z;

% pp = XX * model.W.Mu_W;
% subplot(3,1,1);plot(Z.Mu_Z(end-1000:end));
% subplot(3,1,2);plot([Prec*pp(end-1000:end)  Y(end-1000:end,:)*VPsi']); 
% subplot(3,1,3);plot([pp(end-1000:end)  Y(end-1000:end,:)*model.V.Mu_V']); 
% [mean( (pp-Z.Mu_Z).^2) mean(mean( (Y-Z.Mu_Z*model.V.Mu_V).^2)) ]
% [sum(sum(model.W.Mu_W)) sum(sum(model.V.Mu_V))]
% [ model.Omega.Gam_rate./model.Omega.Gam_shape mean(model.Psi.Gam_rate./model.Psi.Gam_shape) ]
% 
% pause(0.1);
% 



