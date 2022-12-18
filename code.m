clear all; clc; 
load('s195_ses2_task3_final.mat') % subject 195 
artifact2 = sum(artifact');
artifact1 = sum(artifact); 
goodtrial = find(artifact1 < 20);
goodchan =  find(artifact2 < 20); 
erp = squeeze(mean(data(:,:,goodtrial),3)); % mean of all trials, (timepoint)*(channels)

%% I. Bandpass filter - beta 12~30Hz using firls()
n = 360;
% 3 times lower frequency bound = 120ms(from 12Hz) * 3 = 360ms
% 1 timepoint = 1ms from the data, thus 360ms = 360 timepoints 

f = [0 0.0192 0.024 0.06 0.072 1];
% 0.024 for 12Hz, 0.06 for 30Hz where Nyquist frequency = 500Hz (sr = 1000Hz)
% transition zones are marked by 20% of each bounds (0.0048, 0.012) 

a = [0 0 1 1 0 0];

beta = firls(n, f, a); 
% filters out 12~30Hz bands (beta waves) 
% this is equal to the filter kernel in the time domain

%% Bandpass filter - gamma 30~50Hz using firls()
n = 360;
% 3 times lower frequency bound = 120ms(from 12Hz) * 3 = 360ms
% 1 timepoint = 1ms from the data, thus 360ms = 360 timepoints 

f = [0 0.048 0.06 0.1 0.12 1];
% 0.06 for 30Hz, 0.1 for 50Hz where Nyquist frequency = 500Hz (sr = 1000Hz)
% transition zones are marked by 20% of each bounds (0.012, 0.02) 

a = [0 0 1 1 0 0];

gamma = firls(n, f, a); 
% filters out 12~30Hz bands (beta waves) 
% this is equal to the filter kernel in the time domain

%% checking code - apply to test signal 
fs = 800;
[time, simEEG] = simulateEEG(fs*5,fs);

a = sin(2*pi*5*time);
b = sin(2*pi*8*time);
c = sin(2*pi*15*time);
d = sin(2*pi*20*time);
e = sin(2*pi*30*time);
f = sin(2*pi*40*time);
g = sin(2*pi*70*time);
merge = [a;b;c;d;e;f;g]';
   
for i = 1:7
    wave = merge(:,i);
    result = filtfilt(beta, 1, wave); 
    
    figure(1)
    subplot(7,1,i)
    plot(wave([1:2000],:))
    hold on;
    plot(result([1:2000],:))
end 

for i = 1:7
    wave = merge(:,i);
    result = filtfilt(gamma, 1, wave); 
    
    figure(2)
    subplot(7,1,i)
    plot(wave([1:2000],:))
    hold on;
    plot(result([1:2000],:))
end 
   
%% apply bandpass filter 
beta_erp = filtfilt(beta, 1, erp);
figure(1);
plot(erp)
hold on;
plot(beta_erp, 'r')
   title('Beta wave, Filter = 12~30Hz')
   xlabel('Time (1ms)')
   ylabel('Amplitude (mV)')
   
gamma_erp = filtfilt(gamma, 1, erp);
figure(2);
plot(erp)
hold on;
plot(gamma_erp, 'r')
   title('Gamma wave, Filter = 30~50Hz')
   xlabel('Time (1ms)')
   ylabel('Amplitude (mV)')

%% Hilbert transform of filtered data (beta)
h_beta = hilbert(beta_erp);
phase_beta = angle(h_beta);
amp_beta = abs(h_beta);

figure(1)
   subplot(2,1,1);
   plot(amp_beta)
   title('Amplitude of beta wave')
   xlabel('Time (ms)')
   ylabel('Amplitude (mV)')
figure(1) 
   subplot(2,1,2);
   plot(phase_beta)
   title('Phase of beta wave')
   xlabel('Time (ms)')
   ylabel('Angle (degrees)')
   
%% Hilbert transform of filtered data (gamma)
h_gamma = hilbert(gamma_erp);
phase_gamma= angle(h_gamma);
amp_gamma = abs(h_gamma);

figure(1)
   subplot(2,1,1);
   plot(amp_gamma)
   title('Amplitude of gamma wave')
   xlabel('Time (ms)')
   ylabel('Amplitude (mV)')
figure(1) 
   subplot(2,1,2);
   plot(phase_gamma)
   title('Phase of gamma wave')
   xlabel('Time (ms)')
   ylabel('Angle (degrees)')

% Result: mean of beta amplitude slightly bigger than mean gamma amplitude

%% II. PCA of ERP
%% checking PCA code - apply to test signal 
% building the test signals, using sine wave 
fs = 1000; 
N = 0.4*fs; % length of the signal, 400ms
t = [1:N]/fs; % time vector

freq1 = 2; % low freq
freq2 = 10; % high freq

% make different level of power correlation 
% the first pair - high corr 
amp1a = 1; 
amp1b = 0.5;
amp1c = 0.2; 

% the second pair - med corr
amp2a = 0.5;
amp2b = 0.2;
amp2c = 0.8; 

% the third pair - low corr
amp3a = 0.2; 
amp3b = 0.1; 
amp3c = 1; 

sig1 = zeros(10,400);
sig2 = zeros(10,400);
sig3 = zeros(10,400);
for i = 1:10 % 10 randomization each 
    sig1(i,:) = amp1a*sin(2*pi*t*freq1) + amp1b*sin(2*pi*t*freq2) + amp1c * rand(1,N);
    sig2(i,:) = amp2a*sin(2*pi*t*freq1) + amp2b*sin(2*pi*t*freq2) + amp2c * rand(1,N);
    sig3(i,:) = amp3a*sin(2*pi*t*freq1) + amp3b*sin(2*pi*t*freq2) + amp3c * rand(1,N);
end 

sig_merged = [sig1; sig2; sig3];

%% calculating PCA (summarizing into function needed) for merged
sig_merged = sig_merged - repmat(mean(sig_merged,2),1,400);
cov_sig_merged = (sig_merged*sig_merged')./(400-1); 
[pc_sig_merged,eigvals_sig_merged] = eig(cov_sig_merged);
pc_sig_merged = pc_sig_merged(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_sig_merged = diag(eigvals_sig_merged);
time_sig_merged = pc_sig_merged'*sig_merged;
pca_sig_merged = abs(hilbert(time_sig_merged)); % power

%% pca inside each corr groups 
sig1 = sig1 - repmat(mean(sig1,2),1,400);
cov_sig1 = (sig1*sig1')./(400-1); 
[pc_sig1,eigvals_sig1] = eig(cov_sig1);
pc_sig1 = pc_sig1(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_sig1 = diag(eigvals_sig1);
time_sig1 = pc_sig1'*sig1;
pca_sig1 = abs(hilbert(time_sig1)); % power

sig2 = sig2 - repmat(mean(sig2,2),1,400);
cov_sig2 = (sig2*sig2')./(400-1); 
[pc_sig2,eigvals_sig2] = eig(cov_sig2);
pc_sig2 = pc_sig2(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_sig2 = diag(eigvals_sig2);
time_sig2 = pc_sig2'*sig2;
pca_sig2 = abs(hilbert(time_sig2)); % power

sig3 = sig3 - repmat(mean(sig3,2),1,400);
cov_sig3 = (sig3*sig3')./(400-1); 
[pc_sig3,eigvals_sig3] = eig(cov_sig3);
pc_sig3 = pc_sig3(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_sig3 = diag(eigvals_sig3);
time_sig3 = pc_sig3'*sig3;
pca_sig3 = abs(hilbert(time_sig3)); % power

%% PCA - plotting test example
figure();
plot([1:400],sig1(1,:))
hold on;
plot([1:400],sig2(1,:))
hold on;
plot([1:400],sig3(1,:))
hold on;
plot([1:400],pca_sig_merged([1:3],:))
   title('PCA, merged group example')
   xlabel('Time (ms)')
   ylabel('Amplitude (mV)')
   legend('1st group', '2nd group', '3rd group', '1st PC', '2nd PC', '3rd PC', 'Location','southeast')

%% plot each corrgroup 
figure();
subplot(3,1,1)
plot([1:400], sig1(1,:))
hold on;
plot([1:400], pca_sig1(1,:), 'r')
   title('PCA, high correlation')
   xlabel('Time (ms)')
   ylabel('Amplitude (mV)')
   legend('1st channel', '1st PC', 'Location','southeast')

subplot(3,1,2)
plot([1:400], sig2(1,:))
hold on;
plot([1:400], pca_sig2(1,:), 'r')
   title('PCA, med correlation')
   xlabel('Time (ms)')
   ylabel('Amplitude (mV)')
   legend('1st channel', '1st PC', 'Location','southeast')

subplot(3,1,3)
plot([1:400], sig3(1,:))
hold on;
plot([1:400], pca_sig3(1,:), 'r')
   title('PCA, low correlation')
   xlabel('Time (ms)')
   ylabel('Amplitude (mV)')
   legend('1st channel', '1st PC', 'Location','southeast')

%% PCA of ERP
% time window settings (stim onset = 1250ms) 
% prestim = 850 ~ 1250ms 
% poststim = 1250 ~ 1650ms (400ms) 

gamma_erp = gamma_erp';
beta_erp = beta_erp';

%% covariance matrix 
gamma_erp_prestim = gamma_erp(:,[850:1249]); % prestimulus 850 ~ 1250ms 
gamma_erp_prestim = gamma_erp_prestim - repmat(mean(gamma_erp_prestim,2),1,400);
cov_gamma_prestim = (gamma_erp_prestim*gamma_erp_prestim')./(400-1); 

beta_erp_prestim = beta_erp(:,[850:1249]);
beta_erp_prestim = beta_erp_prestim - repmat(mean(beta_erp_prestim,2),1,400);
cov_beta_prestim = (beta_erp_prestim*beta_erp_prestim')./(400-1); 

gamma_erp_poststim = gamma_erp(:,[1251:1650]); % poststimulus 1250 ~ 1650ms 
gamma_erp_poststim = gamma_erp_poststim - repmat(mean(gamma_erp_poststim,2),1,400);
cov_gamma_poststim = (gamma_erp_poststim*gamma_erp_poststim')./(400-1); 

beta_erp_poststim = beta_erp(:,[1251:1650]);
beta_erp_poststim = beta_erp_poststim - repmat(mean(beta_erp_poststim,2),1,400);
cov_beta_poststim = (beta_erp_poststim*beta_erp_poststim')./(400-1); 

figure; imagesc(cov_beta_poststim); colorbar

%% eigenvector decomposition 
[pc_beta_prestim,eigvals_beta_prestim] = eig(cov_beta_prestim);
pc_beta_prestim = pc_beta_prestim(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_beta_prestim = diag(eigvals_beta_prestim);

[pc_beta_poststim,eigvals_beta_poststim] = eig(cov_beta_poststim);
pc_beta_poststim = pc_beta_poststim(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_beta_poststim = diag(eigvals_beta_poststim);

[pc_gamma_prestim,eigvals_gamma_prestim] = eig(cov_gamma_prestim);
pc_gamma_prestim = pc_gamma_prestim(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_gamma_prestim = diag(eigvals_gamma_prestim);

[pc_gamma_poststim,eigvals_gamma_poststim] = eig(cov_gamma_poststim);
pc_gamma_poststim = pc_gamma_poststim(:,end:-1:1); % reverse the order; largest eigenvalues first
eigvals_gamma_poststim = diag(eigvals_gamma_poststim);

%% confirming percentage 
figure; 
subplot(2,2,1)
eigvals1 = 100*eigvals_beta_prestim(end:-1:1)./sum(eigvals_beta_prestim);
plot(eigvals1, 'k.-');
xlabel('Principal component number');
ylabel('Percent variance explained');

subplot(2,2,2)
eigvals2 = 100*eigvals_beta_poststim(end:-1:1)./sum(eigvals_beta_poststim);
plot(eigvals2, 'k.-');
xlabel('Principal component number');
ylabel('Percent variance explained');

subplot(2,2,3)
eigvals3 = 100*eigvals_gamma_prestim(end:-1:1)./sum(eigvals_gamma_prestim);
plot(eigvals3, 'k.-');
xlabel('Principal component number');
ylabel('Percent variance explained');

subplot(2,2,4)
eigvals4 = 100*eigvals_gamma_poststim(end:-1:1)./sum(eigvals_gamma_poststim);
plot(eigvals4, 'k.-');
xlabel('Principal component number');
ylabel('Percent variance explained');

%% calculate time course of each ICAs 
gamma_pcatime_prestim = zeros(10,400);
gamma_pcatime_poststim = zeros(10,400);
beta_pcatime_prestim = zeros(10,400);
beta_pcatime_poststim = zeros(10,400);

for i = 1:10
gamma_pcatime_prestim(i,:) = pc_gamma_prestim(:,i)'*gamma_erp_prestim;
gamma_pcatime_poststim(i,:) = pc_gamma_poststim(:,i)'*gamma_erp_poststim;
beta_pcatime_prestim(i,:) = pc_beta_prestim(:,i)'*beta_erp_prestim;
beta_pcatime_poststim(i,:) = pc_beta_poststim(:,i)'*beta_erp_poststim;
end

%% power over selected 10 electrodes 
gamma_amp_prestim = abs(hilbert(gamma_erp_prestim([9,22,24,36,52,69,89,92,104,124], :)));
gamma_amp_poststim = abs(hilbert(gamma_erp_poststim([9,22,24,36,52,69,89,92,104,124], :)));
beta_amp_prestim = abs(hilbert(beta_erp_prestim([9,22,24,36,52,69,89,92,104,124], :)));
beta_amp_poststim = abs(hilbert(beta_erp_poststim([9,22,24,36,52,69,89,92,104,124], :))); % power (10 electrodes) 

%% power over 10 PCA 
ica_gpre = abs(hilbert(gamma_pcatime_prestim([1:10],:)));
ica_gpos = abs(hilbert(gamma_pcatime_poststim([1:10],:)));
ica_bpre = abs(hilbert(beta_pcatime_prestim([1:10],:)));
ica_bpos = abs(hilbert(beta_pcatime_poststim([1:10],:))); % power (10 ica components) 

%% compare time course power
figure(1); % beta freq
subplot(2,2,1)
plot([850:1249],beta_amp_prestim)
title('Pre-stimulus channel')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

subplot(2,2,2)
plot([850:1249],ica_bpre)
title('Pre-stimulus PCA')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

subplot(2,2,3)
plot([1251:1650],beta_amp_poststim)
title('Post-stimulus channel')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

subplot(2,2,4)
plot([1251:1650],ica_bpos)
title('Post-stimulus PCA')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

figure(2); % gamma freq
subplot(2,2,1)
plot([850:1249],gamma_amp_prestim)
title('Pre-stimulus channel')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

subplot(2,2,2)
plot([850:1249],ica_gpre)
title('Pre-stimulus PCA')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

subplot(2,2,3)
plot([1251:1650],gamma_amp_poststim)
title('Post-stimulus channel')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight

subplot(2,2,4)
plot([1251:1650],ica_gpos)
title('Post-stimulus PCA')
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
axis tight
