%% Synthetic data set made up of sinusoids
clear all

T = 500; max_period = 100; N = 50; noise = 0.5;
% basis sinusoids functions
sinusoids = schauderbasis(max_period,max_period/5,(2*max_period+T)/max_period);

Ntrials = 10;
L = 1; % output components
QQ = 2:5; % number of components
PP = 2:6; % orders

FE = Inf(length(PP),length(QQ));
T = T * ones(Ntrials,1);
ff = 16:35; % waves to consider

signal = noise * randn(sum(T),N);

ww = zeros(length(ff),10);  % loading matrix - which sinusoids participate for each signal
for n=1:N
    while 1
        ww(:,n) = binornd(ones(20,1),0.2);
        if sum(ww(:,n))>1 && sum(ww(:,n))<10
            break
        end
    end
end

for tr=1:Ntrials
    s = zeros(T(tr),N);
    for n=1:N % signal is composed from certain sinusoids + white noise
        for l=1:length(ff)
            phase = round(rand(1)*5);
            s(:,n) = s(:,n) + ww(l,n) * sinusoids(1+phase:phase+T, ff(l));
        end
    end
    tt = sum(T(1:tr-1))+1:sum(T(1:tr));
    signal(tt,:) = signal(tt,:) + s;
end

signal = signal - repmat(mean(signal),size(signal,1),1);
signal = signal ./ repmat(std(signal),size(signal,1),1);

options.cyc = 50;
options.tol = 1e-6;
options.verbose = 0;
options.L = L;

for iQ=1:length(QQ)
    for iP=1:length(PP)
        options.P = 1:PP(iP);
        options.Q = QQ(iQ);
        [m,z,fehist]=lrmartrain(signal,T,options);
        FE(iP,iQ) = fehist(end);
        if fehist(end)==min(FE(:))
            model = m; Z = z; 
        end
        fprintf('Trained model for P=%d,Q=%d \n',PP(iP),QQ(iQ))
    end
end


%% EEG example
% EEG data recorded by Zak Keirn at Purdue University,
% Z. A. Keirn and J. I. Aunon.
% A new mode of communication between man and his surroundings.
% IEEE Transactions on Biomedical engineering, 37:1209?1214, 1990.
clear all

eegdata = dlmread('EEG_example.dat'); eegdata = eegdata';
T = ones(2,1) * size(eegdata,1)/2; % separate it in two trials
eegdata = eegdata - repmat(mean(eegdata),size(eegdata,1),1);
eegdata = eegdata ./ repmat(std(eegdata),size(eegdata,1),1);

options.cyc = 1000;
options.tol = 1e-7;
options.verbose = 1;
options.P = 1:2;
options.Q = 1;
options.L = 1;
options.inittype = 'mar';


[model,Z,fehist]=lrmartrain(eegdata,T,options);

% and plot
figure(1); clf(1)
tt = 1:800;
plot(eegdata(tt+max(options.P),:),'Color',[0.8 0.8 1])
hold on;
plot(Z.Mu_Z(tt,:),'r','LineWidth',2);
plot(mean(eegdata(tt+max(options.P),:),2),'b','LineWidth',2);
[~, y] = pca(eegdata, 'NumComponents', 1 );
plot(y(tt+max(options.P),:),'g','LineWidth',2);
hold off

%% Compare (for the EEG example) different initializations
addpath(genpath('../HMM-MAR'))
clear all

eegdata = dlmread('EEG_example.dat'); eegdata = eegdata';
T = ones(2,1) * size(eegdata,1)/2; % separate it in two trials
eegdata = eegdata - repmat(mean(eegdata),size(eegdata,1),1);
eegdata = eegdata ./ repmat(std(eegdata),size(eegdata,1),1);

options.cyc = 0;
options.tol = 1e-8;
options.verbose = 1;
options.P = 1:10;
options.Q = 1;
options.L = 1;
options.inittype = 'mar'; 
[~,Z1,fehist]=lrmartrain(eegdata,T,options);
options.inittype = 'pca';
[~,Z2,fehist]=lrmartrain(eegdata,T,options);
options.cyc = 1000; options.inittype = 'mar';
[model,Z3,fehist]=lrmartrain(eegdata,T,options);
options.inittype = 'pca';
[model,Z4,fehist]=lrmartrain(eegdata,T,options);


% and plot
figure(1); clf(1)
tt = 1:800;
plot(eegdata(tt+max(options.P),:),'Color',[0.8 0.8 1])
hold on;
plot(Z1.Mu_Z(tt,:),'r','LineWidth',2); % mar
plot(Z2.Mu_Z(tt,:),'b','LineWidth',2); % pca
plot(Z3.Mu_Z(tt,:),'g','LineWidth',2); % lrmar(ini=mar)
plot(Z4.Mu_Z(tt,:),'m','LineWidth',2); % lrmar(ini=pca)
hold off
%%
options_multitaper = struct();
options_multitaper.Fs = 200;
options_multitaper.fpass = [1 15];
options_multitaper.win = 250;
options_multitaper.to_do = [1 0]; % no PDC

fit1 = hmmspectramt(Z1.Mu_Z,size(Z1.Mu_Z,1),options_multitaper);  
fit2 = hmmspectramt(Z2.Mu_Z,size(Z2.Mu_Z,1),options_multitaper);  
fit3 = hmmspectramt(Z3.Mu_Z,size(Z3.Mu_Z,1),options_multitaper);  
fit4 = hmmspectramt(Z4.Mu_Z,size(Z4.Mu_Z,1),options_multitaper);  
fit = hmmspectramt(eegdata,T,options_multitaper);  


figure(2); clf(2)
hold on;
psd = zeros(size(fit.state(1).psd,1),1);
for j=1:size(eegdata,2)
    psd = psd + fit.state(1).psd(:,j,j);
    plot(fit.state(1).f,fit.state(1).psd(:,j,j),'r','LineWidth',1,'Color',[0.8 0.8 1]);
end
plot(fit.state(1).f,psd/size(eegdata,2),'r','LineWidth',5,'Color',[0.8 0.8 1]);
plot(fit1.state(1).f,fit1.state(1).psd(:,1,1),'r','LineWidth',2); % mar
plot(fit2.state(1).f,fit2.state(1).psd(:,1,1),'b','LineWidth',2); % pca
plot(fit3.state(1).f,fit3.state(1).psd(:,1,1),'g','LineWidth',2); % lrmar(ini=mar)
plot(fit4.state(1).f,fit4.state(1).psd(:,1,1),'m','LineWidth',2); % lrmar(ini=pca)
xlim([fit1.state(1).f(1) fit1.state(1).f(end)])
hold off
