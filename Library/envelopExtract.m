function z = envelopExtract(x) 
%
% returns constructed analytic signal
% x is a real-valued record of length N, where N is even %returns the analytic signal z[n]

    x = x(:); %serialize
    N = length(x);
    
    %%% Part 1. z = hilbert(x)
    %
    % FFT of x
    X = fft(x, N);      %YOUR CODE
    half_N = floor(N/2);
    % Create P[m]=Z[m]  from m=1 to N
    P = zeros(size(X));
    P(1) = X(1);
    P(2:half_N) = 2 * X(2:half_N);
    P(half_N + 1) = X(half_N + 1);
    P(half_N + 2 : N) = 0;
     
    % Create z(t)=Zr+j(Zi) from ifft(P)
    z = ifft(P);
    
    % Part 2. Envelope extraction
    inst_amplitude = abs(z);
    
    z = inst_amplitude;

end