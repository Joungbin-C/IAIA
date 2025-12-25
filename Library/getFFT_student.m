function P1=getFFT_student(x,L)
% Returns single-sided spectrum of vector X
% input:    x, 1-d vector
% output:   xFeature, table form


%%% Apply FFT of x
Y = fft(x,L);

half = floor(L/2);
%%% Compute the two-sided spectrum P2
% Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
% Zero frequency P1(0), and the Nyquist frequency P(end) do not occur twice. 
P2 = abs(Y/L);
P1 = P2(1:half+1);

P1(2:half) = 2*P1(2:half);