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
sum_rms = sum(x.*x)/N;
sqrt_rms = sqrt(sum_rms);

xfeature.rms=sqrt_rms;
%% Square Root Average

xfeature.sra = sqrt(mean(sqrt(abs(x))));

%% Average of Absolute Value
xfeature.aav=sum(abs(x))/N;



%% Energy (sum of power_2)
xfeature.energy=sum(x.^2);


%% Peak
xfeature.peak=max(abs(x));


%% Peak2Peak
xfeature.ppv=peak2peak(x);


%% Impulse Factor
xfeature.if=xfeature.peak/xfeature.aav;

%% Shape Factor
xfeature.sf=xfeature.rms/xfeature.aav;

%% Crest Factor
xfeature.cf=xfeature.peak/xfeature.rms;

%% Marginal(Clearance) Factor
xfeature.mf = xfeature.peak / xfeature.sra;

%% Skewness
xfeature.sk=skewness(x);


%% Kurtosis
xfeature.kt=kurtosis(x);

%% Kurtosis Factor
xfeature.kf = xfeature.kt / xfeature.sf^2;

end


