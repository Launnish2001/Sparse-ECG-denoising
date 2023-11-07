clc
clear 

% Signal Database (MIT-BIH Arrhythmia - MITDB)
load("100m.mat") % MITDB Signals (Record - 100,101,103,105,106,115,220)
pure_signal = val(2200:3500); % MITDB ECG signal
t = linspace(0,10,length(pure_signal)); % Time vector
noise = awgn(pure_signal,-20); % Additive White Gaussian Noise(-20dB)
noisy_signal = noise/max(noise); % Noisy Signal (Normalized)
N = length(pure_signal);     % N: signal length
Normalize = mean(abs(pure_signal))/35; % Normalize SNR 

figure;
subplot(2,1,1)
plot(t,pure_signal/max(val),'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Pure Signal")
grid on

subplot(2,1,2)
plot(t,noisy_signal,'r')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Noisy Signal")
grid on

%{

Block Diagram of denoising process:
       _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
 _ _ _|_ _ _                           +|y  
|    (y)    |        _ _ _ _ _         _↓_        _ _ _ _ _ _ _ _   
|Pure_Signal|       |  (LPF)  |   l'_ |   | dw   |     (HPF)     | 
|     +     |------>| Lowpass |---∙-->| ∑ |----->|Sparse Recovery|
|   Noise   |       |_ _ _ _ _|   |   |_ _|      |_ _ _ _ _ _ _ _|
|_ _ _ _ _ _|                     |                     +|d'  
                                  |                     _↓_
                                  ↓                 l'+|   |
                                  -------------------->| ∑ |----> s'= l'+ d'
                                                       |_ _|   
                                                      
Parameters:                                            
y = pure_signal + noise 
l'= Lowpass(y) (Low pass filtered signal)
dw = y - l' (Residual signal)
d' = Sparse Recovery (y) (High pass filtered)
Sparse Recovery Filters - {GMC, L1_Norm, SASS}
s' = l'+ d'(Denoised signal)

GMC - Generalized Minimax Concave Penalty
L1_Norm - Lasso Penalty
SASS - Sparsity assisted signal smoothing (Difference Matrix penalty)

%}

% STFT Matrix
truncate = @(x, N) x(1:N);
AH = @(x) STFT(x,32,2,1,64); % STFT (Time --> Frequency domain)
A = @(X) iSTFT(X,32,2,1,N); % iSTFT (Frequency --> Time domain)

% Lowpass Filter
d_lpf = 2; % degree of filter is 2d
Fc_lpf = 0.03; % cut-off frequency (Normalized by 0.03x360Hz=10.8Hz)
K_lpf = 2; % Order of difference matrix 

[A_l, B_l] = Lowpass(d_lpf, Fc_lpf, N,K_lpf);  % Filter banded matrices 
H =  (A_l^-1)*(B_l); % Highpass sparse matrix
L =  eye(N)-H; % Zero-phase butterworth filter
lpf = L*noisy_signal'; % {l'} Lowpass filtered signal
residual = noisy_signal - lpf'; % {dw} residual component

figure;
subplot(2,1,1)
plot(t,lpf,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Lowpass Signal")
grid on

subplot(2,1,2)
plot(t,residual,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Residual Signal")
grid on

% Highpass sparse filters

% GMC
lambda_GMC = 0.1; % Regularisation parameter
gamma = 0.8; % Convexity parameter
[hpf_GMC, ~, loss_GMC] = GMC(residual, A, AH, 1, lambda_GMC, gamma); % GMC Loss function
hpf_GMC = A(hpf_GMC); % Sparse recovery {d'}
GMC_filtered = (lpf + hpf_GMC)'; % GMC filtered signal

Error_GMC = noisy_signal - GMC_filtered; % Error GMC
RMSE_GMC = sqrt(mean(Error_GMC.^2)) % RMSE GMC
SNR_GMC = snr(Normalize*GMC_filtered,noisy_signal) % SNR GMC

% L1 Norm
lambda_L1Norm = 0.1; % Regularisation parameter
[hpf_L1Norm, ~, loss_L1Norm] = L1NORM(residual, A, AH, 1, lambda_L1Norm); % L1Norm Loss function
hpf_L1Norm = A(hpf_L1Norm); % Sparse recovery {d'}
L1_NORM_filtered = (lpf + hpf_L1Norm)'; % L1_NORM filtered signal

Error_L1Norm = noisy_signal - L1_NORM_filtered; % Error L1Norm
RMSE_L1Norm = sqrt(mean(Error_L1Norm.^2)) % RMSE L1Norm
SNR_L1Norm = snr(Normalize*L1_NORM_filtered,noisy_signal) % SNR L1Norm

% SASS
d_hpf = 3; % degree of filter is 2d
Fc_hpf = 0.032; % cut-off frequency (11.52Hz)
K_hpf = 3; % Order of difference matrix 
lambda_SASS = 1.15; % Regularisation parameter
[x,u,c,hpf_SASS,loss_SASS] = SASS(residual,d_hpf,Fc_hpf,K_hpf,lambda_SASS); % SASS Loss function
hpf_SASS; % Sparse recovery {d'}
SASS_filtered = (lpf + hpf_SASS)'; % SASS filtered signal

Error_SASS = noisy_signal - SASS_filtered; % Error SASS
RMSE_SASS = sqrt(mean(Error_SASS.^2)) % RMSE SASS
SNR_SASS = snr(Normalize*SASS_filtered,noisy_signal)  % SNR SASS

% Comparison of Sparse Recovered Signal
figure;
plot(t,hpf_GMC,'b')
hold on
plot(t,hpf_L1Norm,'r')
plot(t,hpf_SASS,'m')
hold off
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Sparse Recovery")
legend("GMC","L1 Norm","SASS",'Location','best')
grid on 

% Comparison of filtered signal
figure;
subplot(3,1,1)
plot(t,GMC_filtered,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("GMC Filtered")
grid on 
subplot(3,1,2)
plot(t,L1_NORM_filtered,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("L1 Norm Filtered")
grid on 
subplot(3,1,3)
plot(t,SASS_filtered,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("SASS Filtered")
grid on 

% Comparison of Loss function
i = 1:1:length(loss_SASS); % Iterations 
figure;
plot(i,loss_GMC(1:length(i)),'b');yscale("log")
hold on
plot(i,loss_L1Norm(1:length(i)),'r');yscale("log")
plot(i,loss_SASS,'m');yscale("log")
hold off
xlabel("Iterations")
ylabel("Loss")
title("Loss Function")
legend("GMC","L1 Norm","SASS")
grid on

% Comparison of Smoothing 
figure;
subplot(1,2,1)
plot(t,pure_signal/max(val),'b');xlim([5.5 7]);
xlabel("Time (s)")
ylabel("Mag (mV)")
title("MITDB-100 Signal")
grid on

subplot(1,2,2)
plot(t,GMC_filtered,'b');xlim([5.5 7]);
hold on
plot(t,L1_NORM_filtered,'r');xlim([5.5 7]);
plot(t,SASS_filtered,'m');xlim([5.5 7]);
hold off
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Smoothing")
legend("GMC","L1 Norm","SASS",'Location','east')
grid on

% Comparison of RMSE
RMSE = [0.0524 0.0585 0.0631; 0.0387 0.0454 0.0495; 0.0283 0.0368 0.0408; 0.0414 0.0473 0.0513; 0.0303 0.0389 0.0431; 0.0298 0.0391 0.0442; 0.0348 0.0436 0.0528];
Record_No = 1:7;
figure;
bar(Record_No,RMSE)
xticklabels(gca,{'100','101','103','105','106','115','220'});
xlabel("MITDB Record No")
ylabel("Error")
title("RMSE")
legend("GMC","L1 Norm","SASS")
grid on

% Comparison of SNR
SNR = [5.9741 5.8017 5.7044;6.8140 6.5417 6.4178;5.7893 5.2373 4.9017;6.3710 6.0741 5.8062;7.5229 7.0789 6.8251;5.8719 5.5699 5.4353;6.1714 5.9669 5.8230];
figure;
bar(Record_No,SNR)
xticklabels(gca,{'100','101','103','105','106','115','220'});
xlabel("MITDB Record No")
ylabel("dB")
legend("GMC","L1 Norm","SASS")
title("SNR")
grid on
