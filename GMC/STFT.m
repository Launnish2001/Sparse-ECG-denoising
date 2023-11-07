function X = STFT(x,R,M,K,Nfft)


x = x(:).';                             % ensure x is row vector
n = 0:1:R-1;
win  = sin(pi*(n+0.5)/R).^K;                  % cosine window
NC = sqrt(sum(win.^2) * M * Nfft/R);    % normalization constant
x = [zeros(1,R) x zeros(1,R)];          % to deal with first and last block
X = buffer(x,  R, R*(M-1)/M, 'nodelay');
X = bsxfun(@times, win', X);
X = fft(X, Nfft)/NC;                    % FFT applied to each column of X