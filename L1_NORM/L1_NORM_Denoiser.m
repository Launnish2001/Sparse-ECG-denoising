clc
clear 

% Signal Database (MIT-BIH Arrhythmia - MITDB)
load("100m.mat") % Record-100 signal of mitdb is chosen for analysis
pure_signal = val(2200:3500); % MITDB ECG signal
t = linspace(0,10,length(pure_signal)); % Time vector
noise = awgn(pure_signal,-20); % Additive White Gaussian Noise(-20dB)
noisy_signal = noise/max(noise); % Noisy Signal (Normalized)

%{ 
Signal flow model
y = pure_signal + noisy_signal + sparse signal
lpf = Lowpass(y) 
residual = y - lpf
hpf = L1_NORM(residual)
Filtered = lpf(lowpass component) + hpf(Sparse component)
%}

N = length(pure_signal);     % N: signal length

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


% L1_NORM Recovery of highpass components
lambda = 0.1; % Regularisation parameter

%{
Outputs:
hpf = L1_NORM-sparse signal
loss = loss function 
%}

% L1_NORM - FBS Algorithm
[hpf, ~, loss] = L1NORM(residual, A, AH, 1, lambda);
hpf = A(hpf);

figure;
plot(t,hpf,'b')
title("L1_NORM Sparse Signal")

% GMC filtered signal
L1_NORM_filtered = (lpf + hpf)'; % L1_NORM filtered signal

figure;
subplot(3,1,1)
plot(t,pure_signal/max(val),'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("MITDB-100 Signal")
grid on

subplot(3,1,2)
plot(t,noisy_signal,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Noisy Signal")
grid on

subplot(3,1,3)
plot(t,L1_NORM_filtered,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("L1-NORM Filtered Signal")
grid on


% Loss Function Convergence
i = 1:1:length(loss); % Iterations of the algorithm
figure;
plot(i,loss);yscale("log")
xlabel("Iterations")
ylabel("Loss")
title("Loss Function")
grid on

% Resolution Analysis
% STFT of L1_NORM filtered signal 
figure;
subplot(3,1,1)
stft(pure_signal,360,Window=hann(20),OverlapLength=19,FFTLength=2000,FrequencyRange="onesided");
title("MITDB-100 Signal")
subplot(3,1,2)
stft(noisy_signal,360,Window=hann(20),OverlapLength=19,FFTLength=2000,FrequencyRange="onesided");
title("Noisy Signal")
subplot(3,1,3)
stft(L1_NORM_filtered,360,Window=hann(20),OverlapLength=19,FFTLength=2000,FrequencyRange="onesided");
title("L1-NORM Filtered Signal")


% L1-NORM Smoothing
figure;
subplot(1,2,1)
plot(t,pure_signal/max(val),'b');xlim([5.5 7]);
xlabel("Time (s)")
ylabel("Mag (mV)")
title("MITDB-100 Signal")
grid on

subplot(1,2,2)
plot(t,L1_NORM_filtered,'b');xlim([5.5 7]);
xlabel("Time (s)")
ylabel("Mag (mV)")
title("L1-NORM Filtered Signal")
grid on

% Performance Parameters

% RMSE (Root mean square error)
Error = noisy_signal - L1_NORM_filtered;
RMSE = sqrt(mean(Error.^2))

% SNR (Signal to noise ratio)
Normalize = min(abs(pure_signal));
SNR = snr(Normalize*L1_NORM_filtered,noisy_signal) + " dB"

