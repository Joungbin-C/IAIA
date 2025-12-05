function feat_env = extract_envelope_features(DE, FE, fs, rpm)
% Inputs:
%   DE, FE : vibration signals (time-domain)
%   fs     : sampling frequency
%   rpm    : rotational speed (from dataset)
%
% Output:
%   feat_env : 1×72 vector

    Nb = 9;              % number of balls
    Bd = 0.3125;         % ball diameter (inch)
    Pd = 1.462;          % pitch diameter (inch)
    theta = 0;           % contact angle
    
    fr = rpm / 60;       % rotation frequency
    
    BPFO = (Nb/2) * fr * (1 - Bd/Pd*cos(theta));
    BPFI = (Nb/2) * fr * (1 + Bd/Pd*cos(theta));
    BSF  = Pd/(2*Bd) * fr * (1 - (Bd/Pd*cos(theta))^2);

    def_freqs = [BPFI BPFO BSF];   % [inner, outer, ball spin]

    % 2. High-pass filter (remove low-frequency components)
    hpFilt = designfilt('highpassiir','FilterOrder',4, ...
                        'HalfPowerFrequency',500, ...  % cutoff = 500 Hz
                        'SampleRate',fs);

    DE_f = filtfilt(hpFilt, DE);
    FE_f = filtfilt(hpFilt, FE);

    % 3. Analytic signal via Hilbert transform
    DE_h = hilbert(DE_f);
    FE_h = hilbert(FE_f);

    % 4. Envelope spectrum (magnitude FFT)
    N = length(DE);
    f = (0:N-1)*(fs/N);

    ENV_DE = abs(fft(DE_h));
    ENV_FE = abs(fft(FE_h));

    % 5. Narrowband RMS (1% BW around harmonics 1~6)
    BW = 0.01;     % ±1% BW

    feat_env = [];   % result buffer

    % 센서 조합 2×2 (DE→DE, DE→FE, FE→DE, FE→FE)
    S1 = {DE, FE};
    S2 = {DE_h, FE_h};
    Envs = {ENV_DE, ENV_FE};

    % Mapping DE=1, FE=2 (for 2×2 cross detection)
    for src = 1:2
        for trg = 1:2

            ENV = Envs{src};

            for fi = 1:3   % {BPFI, BPFO, BSF}

                base_f = def_freqs(fi);

                for h = 1:6 % harmonics (1~6)

                    target_f = base_f * h;

                    % narrow band range
                    f_low  = target_f*(1-BW);
                    f_high = target_f*(1+BW);

                    idx = (f>=f_low & f<=f_high);
                    band = ENV(idx);

                    % RMS of narrowband
                    if isempty(band)
                        rms_val = 0;
                    else
                        rms_val = rms(band);
                    end

                    feat_env = [feat_env rms_val];

                end
            end
        end
    end

end
