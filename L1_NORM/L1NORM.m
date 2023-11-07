function [x, cost, delta_x] = L1NORM(y, A, AH, rho, lam)

% x = srls_L1(y, A, AH, rho, lam)
%
% srls_L1: Sparse-Regularized Least Squares with L1 norm penalty
%
% Minimize ||y - A x||_2^2 + lam ||x||_1
%
% INPUT
%   y      : data
%   A, AH  : function handles for A and its conj transpose
%   rho    : rho >= maximum eigenvalue of A'A
%   lam    : regularization parameter, lam > 0
%
% OUTPUT
%   x      : solution
%
% [x, cost] = srls_L1(...) returns cost function history

% Algorithm: ISTA (forward-backward splitting)

MAX_ITER = 10000;
TOL_STOP = 1e-4;

% soft thresholding for complex data
soft = @(x, T) max(1 - T./abs(x), 0) .* x;

cost = zeros(1, MAX_ITER);            % cost function history

mu = 1.9 / rho;

% Initialization
AHy = AH(y);                                % A'*y
x = AH(zeros(size(y)));
Ax = A(x);

iter = 0;
old_x = x;

delta_x = [inf];

while (delta_x(end) > TOL_STOP) && (iter < MAX_ITER)
    iter = iter + 1;
    
    z = x - mu * ( AH(Ax) - AHy );
    x = soft(z, lam * mu);
    Ax = A(x);
    
    % cost function history
    residual = y - Ax;
    cost(iter) = 0.5 * sum(abs(residual(:)).^2) + lam * sum(abs(x(:))) ;
    
    delta_x(iter) = max(abs( x(:) - old_x(:) )) / max(abs(old_x(:)));
    old_x = x;   
end

cost = cost(1:iter);
