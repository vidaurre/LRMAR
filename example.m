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

%% Simulating one ROI with one set of MAR dynamics plus noise 

% FIGURE 1, LOW COHERENCE: PCA DESTROYS ALL THE FREQ STRUCTURE, MEAN ALRIGHT
% FIGURE 2, HIGH COHERENCE: ALL THE SAME, MORE OR LESS
% FIGURE 3, FLIPPING SIGN, HIGH COHERENCE, MEAN IS FUCKED UP, PCA ALRIGHT

addpath('../HMM-MAR-scripts')
addpath(genpath('../HMM-MAR'))

coherence = 0.25; % in (0,1), defines to which extent the voxels in the parcel share the same freqs
% TRY HERE TWO VALUES, 0.05 FOR LOW AND 0.25 FOR HIGH, FOR EXAMPLE - THERE
% WILL BE ONE FIGURE FOR EACH ONE
flipping = 1; % flip some channels?
nvoxels = 20; % no. of voxels in the parcel
T = 5000;
noise = 0.5; 
sample_rate = 200;

[X,spectr] = gendata(1,sample_rate,T,nvoxels,coherence,flipping);

options.cyc = 1000;
options.tol = 1e-7;
options.verbose = 0;
options.P = 1:10;
options.Q = 1;
options.L = 1;
options.estimate_V = 0; % what about setting it to 1? same results?

addpath(genpath('../LRMAR'))
[m,z]=lrmartrain(X,size(X,1),options);
rmpath(genpath('../LRMAR'))
s_LRMAR = z.Mu_Z; 
[~,s_PCA] = pca(X,'NumComponents', 1 );
s_mean = mean(X,2);

s_LRMAR = s_LRMAR - mean(s_LRMAR); s_LRMAR = s_LRMAR / std(s_LRMAR);
s_PCA = s_PCA - mean(s_PCA); s_PCA = s_PCA / std(s_PCA);
s_mean = s_mean - mean(s_mean); s_mean = s_mean / std(s_mean);

opt.Fs = sample_rate;
opt.fpass = [1 100];
opt.win = 500;
opt.to_do = [1 0]; % no PDC

psd_LRMAR = hmmspectramt(s_LRMAR,size(s_LRMAR,1),opt);
psd_LRMAR = psd_LRMAR.state(1).psd;
psd_PCA = hmmspectramt(s_PCA,size(s_PCA,1),opt);
psd_PCA = psd_PCA.state(1).psd;
psd_mean = hmmspectramt(s_mean,size(s_mean,1),opt);
psd_mean = psd_mean.state(1).psd;

figure(1); clf(1)
subplot(2,2,1)
hold on
sum_psd = zeros(length(spectr.f),1);
for j=1:nvoxels
   plot( spectr.f',spectr.psd(:,j,j),'Color',[0.7 0.7 1])
   sum_psd = sum_psd + spectr.psd(:,j,j) / nvoxels;
end
plot( spectr.f',sum_psd,'b','LineWidth',2)
hold off
subplot(2,2,2)
hold on
coh_psd = zeros(length(spectr.f),1);
for j=2:nvoxels
   plot( spectr.f',spectr.coh(:,1,j),'Color',[0.7 0.7 1])
   coh_psd = coh_psd + spectr.coh(:,1,j) / (nvoxels-1);
end
plot( spectr.f',coh_psd,'b','LineWidth',2)
hold off
subplot(2,2,3)
hold on
plot(spectr.f',psd_LRMAR,'b','LineWidth',2);
plot(spectr.f',psd_PCA,'r','LineWidth',2);
plot(spectr.f',psd_mean,'Color',[0,0.7,0],'LineWidth',2);
hold off
subplot(2,2,4)
hold on
plot(s_LRMAR(1:200),'b','LineWidth',2);
plot(s_PCA((1:200) + max(options.P)),'r','LineWidth',2);
plot(s_mean((1:200) + max(options.P)),'Color',[0,0.7,0],'LineWidth',2);
hold off

%% Simulating one ROI with two sets of MAR dynamics plus noise 

addpath('../HMM-MAR-scripts')
addpath(genpath('../HMM-MAR'))


coherence = 0.25; % in (0,1), defines to which extent the voxels in the parcel share the same freqs
% TRY HERE TWO VALUES, 0.05 FOR LOW AND 0.25 FOR HIGH
flipping = 0; % flip some channels?
nvoxels = [10 10 10]; % no. of voxels in the parcel per type of MAR
T = 5000;
noise = 0.5; 
sample_rate = 200;

X2 = []; X3 = [];
[X1,spectr1] = gendata(1,sample_rate,T,nvoxels(1),coherence,flipping);
if nvoxels(2)>0, [X2,spectr2] = gendata(2,sample_rate,T,nvoxels(2),coherence,flipping); end
if nvoxels(3)>0, [X3,spectr3] = gendata(3,sample_rate,T,nvoxels(3),coherence,flipping); end

X = [X1 X2 X3];

options.cyc = 1000;
options.tol = 1e-6;
options.verbose = 0;
options.P = 1:10;
options.Q = 1;
options.L = 1;
options.estimate_V = 0;

figure(1); clf(1)
hold on
sum_psd = zeros(length(spectr1.f),1);
for j=1:nvoxels(1)
   plot( spectr1.f',spectr1.psd(:,j,j),'Color',[0.7 0.7 1])
   sum_psd = sum_psd + spectr1.psd(:,j,j) / nvoxels(1);
end
plot( spectr1.f',sum_psd,'b','LineWidth',2)
if nvoxels(2)>0,
    sum_psd = zeros(length(spectr2.f),1);
    for j=1:nvoxels(2)
        plot( spectr2.f',spectr2.psd(:,j,j),'Color',[1 0.7 0.7])
        sum_psd = sum_psd + spectr2.psd(:,j,j) / nvoxels(2);
    end
    plot( spectr2.f',sum_psd,'r','LineWidth',2)
end
if nvoxels(3)>0,
    sum_psd = zeros(length(spectr3.f),1);
    for j=1:nvoxels(3)
        plot( spectr3.f',spectr3.psd(:,j,j),'Color',[1 0.7 0.7])
        sum_psd = sum_psd + spectr3.psd(:,j,j) / nvoxels(3);
    end
    plot( spectr3.f',sum_psd,'Color',[0 0.7 0],'LineWidth',2)
end
%%
QQ = 1:4;
FE_LRMAR = zeros(length(QQ),1);
%FE_PCA = zeros(length(QQ),1); % this does not 
PSD_LRMAR = cell(length(QQ),1);
for Q = QQ
    % LRMAR
    addpath(genpath('../LRMAR'))
    options.Q = Q;
    [~,z,fe]=lrmartrain(X,size(X,1),options);
    FE_LRMAR(Q) = fe(end);
    rmpath(genpath('../LRMAR'))
    s_LRMAR = z.Mu_Z;
    opt.Fs = sample_rate;
    opt.fpass = [1 100];
    opt.win = 500;
    opt.to_do = [1 0]; % no PDC
    psd_LRMAR = [];
    for q=1:Q
        psd = hmmspectramt(s_LRMAR(:,q),size(s_LRMAR,1),opt);
        psd_LRMAR = [psd_LRMAR psd.state.psd];
    end
    PSD_LRMAR{Q} = psd_LRMAR;
    % Variational Bayesian PCA
    %addpath('../BayesianPCA/')
    %[ A, S,] = pca_full( X, Q, struct( 'maxiters', 10,...
    %    'algorithm', 'vb',...
    %    'uniquesv', 0,...
    %    'cfstop', [ 100 0 0 ],...
    %    'minangle', 0 ) );
    %rmpath('../BayesianPCA/')
end

figure(2);
bar(FE_LRMAR); ylim([0.99*min(FE_LRMAR) 1.01*max(FE_LRMAR) ])
%%
figure(3); clf(3)
Color = {'b','r',[0 0.7 0],'m'};
for q=1:length(QQ)
    subplot(1,length(QQ),q)
    hold on
    for j = 1:size(PSD_LRMAR{q},2)
        plot(spectr.f',PSD_LRMAR{q}(:,j),'Color',Color{j},'LineWidth',2);
    end
    hold off
end
