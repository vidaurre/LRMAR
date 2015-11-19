T = 1000; max_period = 100; N = 50; noise = 0.5;
sinusoids = schauderbasis(max_period,max_period/5,(2*max_period+T)/max_period); % basis sinusoids functions

Ntrials = 100;
L = 1;
QQ = 2:10;
PP=2:6;

FE = zeros(Ntrials,length(PP),length(QQ));

for tr=1:Ntrials
    
    signal = noise * randn(T,N);
    ff = 16:35; % waves to consider
    ww = zeros(length(ff),10);  % loading matrix - which sinusoids participate for each signal
    for n=1:N
        while 1
            ww(:,n) = binornd(ones(20,1),0.2);
            if sum(ww(:,n))>1 && sum(ww(:,n))<10
                break
            end;
        end;
    end;
    for n=1:N % signal is composed from certain sinusoids + white noise
        for l=1:length(ff)
            phase = round(rand(1)*5);
            signal(:,n) = signal(:,n) + ww(l,n) * sinusoids(1+phase:phase+T, ff(l));
        end;
    end;
    
    for iQ=1:length(QQ)
        for iP=1:length(PP)
            options.cyc = 2000;
            options.tol = 0.0001;
            options.showfe = 1;
            options.P = PP(iP);
            options.Q = QQ(iQ);
            options.L = L;
            [model,Z,fehist]=lrmartrain(signal,options);
            FE(tr,iP,iQ) = fehist(end);  
        end;    
    end;
    
end;