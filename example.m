%% Generate a data set made up of sinusoids
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