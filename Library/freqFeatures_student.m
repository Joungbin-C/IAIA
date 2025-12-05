function xfeature=freqFeatures_student(F)
% Returns frequency-domain features of FFT(x)
% input:    F, 1-d vector
% output:   xFeature, table form


% Create table variable xfeature
xfeature = table;
N=length(F);


% Frequency Center
% YOUR CODE GOES HERE
xfeature.fc= sum(F) / N;

% RMS frequency
% YOUR CODE GOES HERE
xfeature.rmsf=sqrt(sum(F.^2) / N);

% Root variance frequency
% YOUR CODE GOES HERE
xfeature.rvf=sqrt(sum((F-xfeature.fc).^2) / N);