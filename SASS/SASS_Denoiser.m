clc
clear all

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
hpf = SASS(residual)
Filtered = lpf(lowpass component) + hpf(Sparse component)
%}

% Lowpass Filter
d_lpf = 3; % degree of filter is 2d
Fc_lpf = 0.03; % cut-off frequency (Normalized by 0.03x360Hz=10.8Hz)
N = length(pure_signal); 
K_lpf = 3; % Order of difference matrix 

[A, B] = Lowpass(d_lpf, Fc_lpf, N,K_lpf);  % Filter banded matrices 
H =  (A^-1)*(B); % Highpass sparse matrix
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


% Sparsity Recovery of highpass components
d_hpf = 3; % degree of filter is 2d
Fc_hpf = 0.032; % cut-off frequency (11.52Hz)
K_hpf = 3; % Order of difference matrix 
lambda = 1.15; % Regularisation parameter

%{
Outputs:
u = SASS-sparse signal
hpf = filtered-sparse signal
loss = loss function 
%}

% SASS - SASS Algorithm 
[x,u,c,hpf,loss] = SASS(residual,d_hpf,Fc_hpf,K_hpf,lambda);

figure;
subplot(2,1,1)
plot(linspace(0,10,length(u)),u,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Sparse Signal")
grid on

subplot(2,1,2)
plot(t,hpf,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("Filtered-Sparse Signal")
grid on


% SASS filtered signal
SASS_filtered = (lpf + hpf)'; % SASS filtered signal

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
plot(t,SASS_filtered,'b')
xlabel("Time (s)")
ylabel("Mag (mV)")
title("SASS Filtered Signal")
grid on

% Loss function Convergence
i = 1:1:length(loss); % Iterations of the algorithm
figure;
plot(i,loss);yscale("log")
xlabel("Iterations")
ylabel("Loss")
title("Loss Function")
grid on

% Resolution Analysis
% STFT of SASS filtered signal shows the high frequency components being retained
figure;
subplot(3,1,1)
stft(pure_signal,360,Window=hann(20),OverlapLength=19,FFTLength=2000,FrequencyRange="onesided");
title("MITDB-100 Signal")
subplot(3,1,2)
stft(noisy_signal,360,Window=hann(20),OverlapLength=19,FFTLength=2000,FrequencyRange="onesided");
title("Noisy Signal")
subplot(3,1,3)
stft(SASS_filtered,360,Window=hann(20),OverlapLength=19,FFTLength=2000,FrequencyRange="onesided");
title("SASS Filtered Signal")


% SASS Smoothing 
figure;
subplot(1,2,1)
plot(t,pure_signal/max(val),'b');xlim([5.5 7]);
xlabel("Time (s)")
ylabel("Mag (mV)")
title("MITDB-100 Signal")
grid on

subplot(1,2,2)
plot(t,SASS_filtered,'b');xlim([5.5 7]);
xlabel("Time (s)")
ylabel("Mag (mV)")
title("SASS Filtered Signal")
grid on

% Performance Parameters

% RMSE (Root mean square error)
Error = noisy_signal - SASS_filtered;
RMSE = sqrt(mean(Error.^2))

% SNR (Signal to noise ratio)
Normalize = min(abs(pure_signal));
SNR = snr(Normalize*SASS_filtered,noisy_signal) + " dB"


