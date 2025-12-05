function xfeature=timeFeatures_student(x)
% Returns time-domain features of vector x
% input:    x, 1-d vector
% output:   xFeature in table form


xfeature = table;
N=length(x);



%% mean and STD
xfeature.mean=mean(x);
xfeature.std=std(x);

%% RMS
%**** Your code goes here ****
xfeature.rms=0;
%% Square Root Average
%**** Your code goes here ****
xfeature.sra=0;

%% Average of Absolute Value
xfeature.aav=sum(abs(x))/N;



%% Energy (sum of power_2)
xfeature.energy=sum(x.^2);


%% Peak
%**** Your code goes here ****
xfeature.peak=0

%% Peak2Peak
xfeature.ppv=peak2peak(x);


%% Impulse Factor
xfeature.if=xfeature.peak/xfeature.aav;

%% Shape Factor
xfeature.sf=xfeature.rms/xfeature.aav;

%% Crest Factor
%**** Your code goes here ****
xfeature.cf=0;


% Marginal(Clearance) Factor
xfeature.mf=xfeature.peak/xfeature.sra;

%% Skewness
xfeature.sk=skewness(x);


%% Kurtosis
xfeature.kt=kurtosis(x);



