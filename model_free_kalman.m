%% COMPARISON BETWEEN MODEL-BASED AND MODEL-FREE DYNAMIC FILTERS
% Sean McGowan 3/7/25

clear all
clc

%%
%-------------------------------Parameters---------------------------------

% System parameters
global Fx Fy a_true Nx Ny eps hx hy
a_true = 1;
Fx = 10;
Fy = 0;
Nx = 40;
Ny = 10;
eps = 0.1;
hx = -1;
hy = 1;

% Model parameter perturbation
global a_model
a_model = 1;

% Simulation parameters
global dt
Kinit = 500; % Spinup period
Ktrain = 1000; % Training data - 1000 for L9640, 2000 for L9610
Ktest = 200; % Testing data
dt = 0.05;

% Embedding parameters
global D nu tau
D = 2; % Number of time delays
nu = 2; % Number of spatial steps
Nemb = (2*nu+1)*(D+1);
tau = 4;

% Number of nearest neighbours for interpolation
Nnn = 30;

%-----------------------Generate and process data--------------------------

var_index = 1;

% Generate data from true system
x0 = unifrnd(0,1,1,Nx*(Ny+1));% Randomise initial condition
[~, X_full] = ode45(@(t,X) l96MS(t,X,1), 0:dt:dt*(Kinit+Ktrain+Ktest), x0);
X_true = X_full(Kinit+(1:(Ktrain+Ktest)),:)';
x_true = X_true(1:Nx,:);

% Add measurement noise and mask to observations
variance_vec = [0.1 0.2 0.5 1 2 5];
variance = variance_vec(var_index);
x_noisy = x_true+normrnd(0,sqrt(variance),Nx,Ktrain+Ktest);
x_train = x_noisy(:,1:Ktrain);
x_test = x_noisy(:,1+Ktrain:end);


%%
%---------------------------Remove observations----------------------------

% Random missing obs
%mask = rand(Nx,Ktest+D*tau)>0.5;

% Sparse in time
% mask = zeros(Nx,Ktest+D*tau);
% Tskip = 40;
% mask(:,D*tau+1:Tskip:end) = 1; 
% mask(:,1:Tskip:D*tau+1) = 1;

% Sparse in space
mask = zeros(Nx,Ktest+D*tau);
mask(1:10:end,:) = 1; 

% All data
% mask = ones(Nx,Ktest+D*tau);

% Satellite-like observations
% S = 8;
% mask_sat = repmat(imresize(eye(Nx),[Nx S*Nx],'nearest'),[1 ceil((Ktest+D*tau)/(S*Nx))]);
% mask = mask_sat(:,1:Ktest+D*tau)+circshift(mask_sat(:,1:Ktest+D*tau),[10 0])...
%     +circshift(mask_sat(:,1:Ktest+D*tau),[20 0])...
%     +circshift(mask_sat(:,1:Ktest+D*tau),[30 0]);
% mask = flip(mask);


mask_delay = mask;
mask = mask(:,D*tau+1:end);

y = x_test.*mask;
y_delay = x_noisy(:,1+Ktrain-D*tau:end).*mask_delay;


figure
subplot(1,2,1)
imagesc(x_test)

subplot(1,2,2)
yplot = y;
yplot(yplot==0) = nan;
imagesc(yplot)



%%

%-------------------------Populate embedded vectors------------------------
x_embed = zeros(Nemb,Nx,Ktrain-D*tau);
x_true_embed_1 = zeros(Nemb,Nx);
y_embed = zeros(Nemb,Nx,Ktest);
mask_embed = zeros(Nemb,Nx,Ktest);
mask_embed_equivalent = zeros(Nemb,Nx);
NU = [0 -nu:-1 1:nu];
count = 0;
for n = 1:length(NU)   
    for d = 0:D
        count = count+1;
        % Noisy training data
        x_embed(count,:,:) = ...
            circshift(x_train(:,1+(D-d)*tau:end-d*tau),NU(n),1);
        % Initial embedded ground truth for initialisation
        x_true_embed_1(count,:) = ...
            circshift(x_true(:,1+Ktrain-d*tau),NU(n),1);
        % Observations
        y_embed(count,:,:) = ...
            circshift(y_delay(:,1+(D-d)*tau:end-d*tau),NU(n),1);
        % Mask
        mask_embed(count,:,:) = ...
            circshift(mask_delay(:,1+(D-d)*tau:end-d*tau),NU(n),1);
    end
    if NU(n)>=0
        mask_embed_equivalent((0:D)+1+(n-1)*(D+1),1+NU(n)) = 1;
    else
        mask_embed_equivalent((0:D)+1+(n-1)*(D+1),Nx+1+NU(n)) = 1;
    end
end



%%
%--------------------------------Initialise--------------------------------
% Noise covariance matrices - process noise Q, observation noise R
Q = 0.5*eye(Nx); % Process
Q_embed = 0.5*eye(Nemb);
R = variance*eye(Nx); % Observation
R_embed = variance*eye(Nemb);
% Observation function
H = eye(Nx);
H_emb = eye(Nemb);

%----------------------------------States----------------------------------
% Model-based Extended Single scale
x_ex_mb_a = zeros(Nx,Ktest);
x_ex_mb_a(:,1) = x_true(:,1+Ktrain);
% Model-free Extended
x_ex_mf_a = zeros(Nemb,Nx,Ktest);
x_ex_mf_a(:,:,1) = x_true_embed_1;
% Model-based Ensemble Single scale
Nens_mb = 100; % Ensemble size
mu_inf_mb = 1.02; % Inflation -  mu_inf = 1 for no inflation
x_en_mb_a_bar = zeros(Nx,Ktest);
u1 = normrnd(0,sqrt(variance),Nx,Nens_mb);
u1 = u1-mean(u1,2);
x_en_mb_a = repmat(x_true(:,1+Ktrain),1,Nens_mb)+u1;
x_en_mb_a_bar(:,1) = x_true(:,1+Ktrain);
% Model-free Ensemble
Nens_mf = 100;
mu_inf_mf = 1;
x_en_mf_a_bar = zeros(Nemb,Nx,Ktest);
u1_embed = normrnd(0,sqrt(variance),Nemb,Nx,Nens_mf);
u1_embed = u1_embed-mean(u1_embed,3);
x_en_mf_a = repmat(x_true_embed_1,1,1,Nens_mf)+u1_embed;
x_en_mf_a_bar(:,:,1) = x_true_embed_1;
% Model-based Unscented Single scale
alpha_un = 0.01;
beta_un = 2;
kappa_un = 0;
lambda_un = alpha_un^2*(Nx+kappa_un)-Nx;
x_un_mb_a = zeros(Nx,Ktest);
x_un_mb_a(:,1) = x_true(:,1+Ktrain);
Wm = [lambda_un/(lambda_un+Nx) (0.5/(lambda_un+Nx))*ones(1,2*Nx)];
Wc = Wm;
Wc(1) = Wc(1)+1-alpha_un^2+beta_un;
% Model-free Unscented
alpha_un_embed = 1;
x_un_mf_a = zeros(Nemb,Nx,Ktest);
x_un_mf_a(:,:,1) = x_true_embed_1;
lambda_un_embed = alpha_un_embed^2*(Nemb+kappa_un)-Nemb;
Wm_embed = [lambda_un_embed/(lambda_un_embed+Nemb)...
    (0.5/(lambda_un_embed+Nemb))*ones(1,2*Nemb)];
Wc_embed = Wm_embed;
Wc_embed(1) = Wc_embed(1)+1-alpha_un_embed^2+beta_un;
% Redundancy update - gamma=1 values equal mean, gamma=0 values don't change
gamma = 0.9;

%-------------------------Error covariance matrices------------------------
% Model-based Extended Single scale
P_ex_mb_a = R;
% Model-free Extended
P_ex_mf_a = repmat(R_embed,1,1,Nx);
% Model-based Unscented Single scale
P_un_mb_a = R;
% Model-free Unscented
P_un_mf_a = repmat(R_embed,1,1,Nx);

% Instability catch variables
ex_mb = 0;
en_mb = 0;
un_mb = 0;
ex_mf = 0;
en_mf = 0;
un_mf = 0;


% Loop through observations
for kk = 2:Ktest
    kk
%---------------------------------Forecast---------------------------------
    % Model-based Extended Single scale
    if ex_mb == 0
        [x_ex_mb_f,P_ex_mb_f] = ex_mbss_f(x_ex_mb_a(:,kk-1),P_ex_mb_a,Q);
    end
    % Model-free Extended
    if ex_mf == 0
    [x_ex_mf_f,P_ex_mf_f] = ex_mf_f(x_ex_mf_a(:,:,kk-1),P_ex_mf_a,x_embed,...
        Nnn,Q_embed);
    end
    % Model-based Ensemble Single Sscale
    if en_mb == 0
    x_en_mb_f = en_mbss_f(x_en_mb_a,Q);
    end
    % Model-free Ensemble
    if en_mf == 0
    x_en_mf_f = en_mf_f(x_en_mf_a,x_embed,Nnn,Q_embed);
    end
    % Model-based Unscented Single scale
    if un_mb == 0
    [x_un_mb_f,P_un_mb_f] = u_mb_f(x_un_mb_a(:,kk-1),P_un_mb_a,...
        lambda_un,Wm,Wc,Q);
    end
    % Model-free Unscented
    if un_mf == 0
    [x_un_mf_f,P_un_mf_f] = u_mf_f(x_un_mf_a(:,:,kk-1),P_un_mf_a,x_embed,...
      Nnn,lambda_un_embed,Wm_embed,Wc_embed,Q_embed);
    end

%---------------------------------Analysis---------------------------------
    % Model-based Extended Single scale
    if ex_mb == 0
    [x_ex_mb_a(:,kk),P_ex_mb_a] = ex_mb_a(x_ex_mb_f,P_ex_mb_f,...
        y(:,kk),diag(mask(:,kk))*H,R);
    end
    % Model-free Extended
    if ex_mf == 0
    [x_ex_mf_a(:,:,kk),P_ex_mf_a] = ex_mf_a(x_ex_mf_f,P_ex_mf_f,y_embed(:,:,kk),...
       bsxfun(@times,reshape(mask_embed(:,:,kk),Nemb,1,Nx),H_emb),R_embed);
    end
    % Model-based Ensemble Single scale
    if en_mb == 0
    x_en_mb_a = en_mb_a(x_en_mb_f,y(:,kk),...
        diag(mask(:,kk))*H,variance,mu_inf_mb);
    x_en_mb_a_bar(:,kk) = mean(x_en_mb_a,2);
    end
    % Model-free Ensemble
    if en_mf == 0
    x_en_mf_a = en_mf_a(x_en_mf_f,y_embed(:,:,kk),...
        bsxfun(@times,reshape(mask_embed(:,:,kk),Nemb,1,Nx),H_emb),...
        variance);
    end
    % Model-based Unscented Single scale
    if un_mb == 0
    [x_un_mb_a(:,kk),P_un_mb_a] = u_mb_a(x_un_mb_f,P_un_mb_f,y(:,kk),...
        diag(mask(:,kk))*H,Wm,Wc,R);
    end
    % Model-free Unscented
    if un_mf == 0
    [x_un_mf_a(:,:,kk),P_un_mf_a] = u_mf_a(x_un_mf_f,P_un_mf_f,y_embed(:,:,kk),...
        bsxfun(@times,reshape(mask_embed(:,:,kk),Nemb,1,Nx),H_emb),...
        Wm_embed,Wc_embed,R_embed);
    end

    % Average equivalent state estimates in embedding matrices
    % only works currently for nu=2, D=2
    % Extended
    if ex_mf == 0
    x_ex_mf_a(:,:,kk) = nudge(x_ex_mf_a(:,:,kk),mask_embed_equivalent,gamma);
    end
    % Ensemble
    if en_mf == 0
    for n = 1:Nens_mf
    x_en_mf_a(:,:,n) = nudge(x_en_mf_a(:,:,n),mask_embed_equivalent,gamma);
    end
    for i = 1:Nx
    xabari = mean(x_en_mf_a,2); % Inflation after redundancy 
    x_en_mf_a(:,i,:) = xabari+mu_inf_mf*(x_en_mf_a(:,i,:)-xabari);
    end
    end
    % Unscented
    if un_mf == 0
    x_un_mf_a(:,:,kk) = nudge(x_un_mf_a(:,:,kk),mask_embed_equivalent,gamma);
    end
    if en_mf == 0
    x_en_mf_a_bar(:,:,kk) = mean(x_en_mf_a,3);
    end


    % Catch instability
    if any(abs(x_ex_mb_a(:,kk))>1000,'all') && ex_mb == 0
        ex_mb = 1;
    end
    if any(abs(x_en_mb_a)>1000,'all') && en_mb == 0
        en_mb = 1;
    end
    if any(abs(x_un_mb_a(:,kk))>1000,'all') && un_mb == 0
        un_mb = 1;
    end
    if any(abs(x_ex_mf_a(:,:,kk))>1000,'all') && ex_mf == 0
        ex_mf = 1;
    end
    if any(abs(x_en_mf_a)>1000,'all') && en_mf == 0
        en_mf = 1;
    end
    if any(abs(x_un_mf_a(:,:,kk))>1000,'all') && un_mf == 0
        un_mf = 1;
    end

end


yplot = y;
yplot(yplot==0) = nan;

figure
subplot(4,2,1)
imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',x_true(:,1+Ktrain:end))
title('True')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,2)
h=imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',yplot);
set(h, 'AlphaData', ~isnan(yplot))
title('Observations')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,3)
h_exkfmb = imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',x_ex_mb_a);
set(h_exkfmb, 'AlphaData', x_ex_mb_a~=0)
title('EKF MB')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,5)
imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',x_en_mb_a_bar)
title('ENKF MB')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,7)
imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',x_un_mb_a)
title('UKF MB')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,4)
imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',squeeze(x_ex_mf_a(1,:,:)))
title('EKF MF')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,6)
imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',squeeze(x_en_mf_a_bar(1,:,:)))
title('ENKF MF')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])
subplot(4,2,8)
imagesc('xdata',(1:200)*dt,'ydata',1:40,'cdata',squeeze(x_un_mf_a(1,:,:)))
title('UKF MF')
clim([-5 10])
xlim([0 10])
ylim([0.5 40.5])


RMSE_ex_mb = sqrt(mean((1./(1:Ktest)).*cumsum((x_true(:,1+Ktrain:end)-...
    x_ex_mb_a).^2,2),1));
RMSE_en_mb = sqrt(mean((1./(1:Ktest)).*cumsum((x_true(:,1+Ktrain:end)-...
    x_en_mb_a_bar).^2,2),1));
RMSE_un_mb = sqrt(mean((1./(1:Ktest)).*cumsum((x_true(:,1+Ktrain:end)-...
    x_un_mb_a).^2,2),1));
RMSE_ex_mf = sqrt(mean((1./(1:Ktest)).*cumsum((x_true(:,1+Ktrain:end)-...
    squeeze(x_ex_mf_a(1,:,:))).^2,2),1));
RMSE_en_mf = sqrt(mean((1./(1:Ktest)).*cumsum((x_true(:,1+Ktrain:end)-...
    squeeze(x_en_mf_a_bar(1,:,:))).^2,2),1));
RMSE_un_mf = sqrt(mean((1./(1:Ktest)).*cumsum((x_true(:,1+Ktrain:end)-...
    squeeze(x_un_mf_a(1,:,:))).^2,2),1));


figure
plot((1:200)*dt,RMSE_ex_mb,'k-')
hold on
plot((1:200)*dt,RMSE_en_mb,'r-')
plot((1:200)*dt,RMSE_un_mb,'b-')
plot((1:200)*dt,RMSE_ex_mf,'k--')
plot((1:200)*dt,RMSE_en_mf,'r--')
plot((1:200)*dt,RMSE_un_mf,'b--')


%% FUNCTIONS
%% Model-based EXKF
% Forecast
% Single scale model
function [xf,Pf] = ex_mbss_f(xa,Pa,Q)
global a_model dt
xf = intdyn(@(t,X) l96SS(t,X,a_model),xa,dt); % Forecast
TLM = expm(l96SS_TLM(0,xa,a_model)*dt); % Tangent linear model
Pf = TLM*Pa*TLM'+Q; % Covariance
end

% Analysis
function [xa,Pa] = ex_mb_a(xf,Pf,y,H,R)
Nx = length(xf);
Kk = Pf*H'*inv(H*Pf*H'+R); % Gain
xa = xf+Kk*(y-H*xf); % Innovation
Pa = (eye(Nx)-Kk*H)*Pf; % Covariance
end

%% Model-free EXKF

% Forecast
function [xf,Pf] = ex_mf_f(xa,Pa,x_embed,Nnn,Q)
Nemb = size(xa,1);
Nx = size(xa,2);
xf = zeros(size(xa));
Pf = zeros(Nemb,Nemb,Nx);
for i = 1:Nx
    xf(:,i) = nearneighM(xa(:,i),x_embed,Nnn); % Forecast
    TLM = embed_TLM(xa(:,i),x_embed,Nnn);
    Pf(:,:,i) = TLM*Pa(:,:,i)*TLM'+Q;
end
end

% Analysis
function [xa,Pa] = ex_mf_a(xf,Pf,y,H,R)
Nemb = size(xf,1);
Nx = size(xf,2);
xa = zeros(size(xf));
Pa = zeros(Nemb,Nemb,Nx);
for i = 1:Nx
    Kk = Pf(:,:,i)*H(:,:,i)'*inv(H(:,:,i)*Pf(:,:,i)*H(:,:,i)'+R); % Gain
    xa(:,i) = xf(:,i)+Kk*(y(:,i)-H(:,:,i)*xf(:,i)); % Innovation
    Pa(:,:,i) = (eye(Nemb)-Kk*H(:,:,i))*Pf(:,:,i); % Covariance
end
end

%% Model-based ENKF
% Forecast
% Single scale model
function xfi = en_mbss_f(xai,Q)
global a_model dt
Nx = size(xai,1);
Nens = size(xai,2);
for i = 1:Nens
    xfi(:,i) = intdyn(@(t,X) l96SS(t,X,a_model),xai(:,i),dt)+...
        mvnrnd(zeros(Nx,1),Q)'; % Forecast
end
end

% Analysis
function xai = en_mb_a(xfi,y,H,variance,mu_inf)
Nens = size(xfi,2);
xfbar = mean(xfi,2);
Pf = 1/(Nens-1)*(xfi-xfbar)*(xfi-xfbar).';
ui = normrnd(zeros(size(xfi)),sqrt(variance)*ones(size(xfi)));
ui = ui - mean(ui,2);
Ru = 1/(Nens-1)*(ui*ui'); % Empirical error covariance matrix
yi = y+ui; % Perturb observations
Kk = Pf*H'*inv(H*Pf*H'+Ru); % Gain
xai = xfi+Kk*(yi-H*xfi); % Innovation
% Inflation
xabar = mean(xai,2);
xai = xabar+mu_inf*(xai-xabar);
end

%% Model-free ENKF
% Forecast
function xfi = en_mf_f(xai,x_embed,Nnn,Q)
Nemb = size(xai,1);
Nx = size(xai,2);
Nens = size(xai,3);
xfi = zeros(size(xai));
for i = 1:Nx
    for j = 1:Nens
        xfi(:,i,j) = nearneighM(xai(:,i,j),x_embed,Nnn)+...
            mvnrnd(zeros(Nemb,1),Q)'; % Forecast
    end
end
end

% Analysis
function xai = en_mf_a(xfi,y,H,variance)
Nemb = size(xfi,1);
Nx = size(xfi,2);
Nens = size(xfi,3);
xai = zeros(size(xfi));
for i = 1:Nx
    xfbari = squeeze(mean(xfi(:,i,:),3));
    Pfi = 1/(Nens-1)*(squeeze(xfi(:,i,:))-xfbari)*(squeeze(xfi(:,i,:))-xfbari).';
    ui = normrnd(zeros(Nemb,Nens),sqrt(variance)*ones(Nemb,Nens));
    ui = ui - mean(ui,2);
    Ru = 1/(Nens-1)*(ui*ui'); % Empirical error covariance matrix
    yi = y(:,i)+ui; % Perturb observations
    Kki = Pfi*H(:,:,i)'*inv(H(:,:,i)*Pfi*H(:,:,i)'+Ru); % Gain
    xai(:,i,:) = squeeze(xfi(:,i,:))+Kki*(yi-H(:,:,i)*squeeze(xfi(:,i,:))); % Innovation
end
end

%% Model-based UKF
% Forecast
% Single scale model
function [xfi,Pf] = u_mb_f(xa,Pa,lambda_un,Wm,Wc,Q)
global a_model dt
Nx = size(xa,1);
xfi = zeros(Nx,2*Nx+1);
sig = [xa repmat(xa,1,Nx)+chol((Nx+lambda_un)*Pa,'lower')', ...
    repmat(xa,1,Nx)-chol((Nx+lambda_un)*Pa,'lower')'];
for i = 1:2*Nx+1 % Forecast sigma points
    xfi(:,i) = intdyn(@(t,X) l96SS(t,X,a_model),sig(:,i),dt); 
end
Pf = Q;
for i = 1:2*Nx+1 % Weighted covariance
    diff = xfi(:, i)-xfi*Wm';
    Pf = Pf+Wc(i)*(diff * diff');
end
end

% Analysis
function [xa,Pa] = u_mb_a(xfi,Pf,y,H,Wm,Wc,R)
Nx = size(xfi,1);

Pyy = R;
Pxy = zeros(Nx);
for i = 1:2*Nx+1
    ydiff = xfi(:,i)-H*xfi*Wm';
    xdiff = xfi(:,i)-xfi*Wm';
    Pyy = Pyy+Wc(i)*(ydiff*ydiff');
    Pxy = Pxy+Wc(i)*(xdiff*ydiff');
end
Kk = Pxy/Pyy; % Gain
xa = xfi*Wm'+Kk*(y-H*xfi*Wm'); % Innovation
Pa = Pf-Kk*Pyy*Kk';
end

%% Model-free UKF
% Forecast
function [xfi,Pf] = u_mf_f(xa,Pa,x_embed,Nnn,lambda_un,Wm,Wc,Q)
Nemb = size(xa,1);
Nx = size(xa,2);
xfi = zeros(Nemb,Nx,2*Nemb+1);
Pf = zeros(Nemb,Nemb,Nx);
for i = 1:Nx % Forecast sigma points
    xai = xa(:,i);
    sig = [xai repmat(xai,1,Nemb)+chol((Nemb+lambda_un)*Pa(:,:,i),'lower')', ...
    repmat(xai,1,Nemb)-chol((Nemb+lambda_un)*Pa(:,:,i),'lower')'];
    for j = 1:2*Nemb+1
        xfi(:,i,j) = nearneighM(sig(:,j),x_embed,Nnn); 
    end
end
Pf = repmat(Q,1,1,Nx);
for i = 1:Nx % Weighted covariance
    for j = 1:2*Nemb+1
        diff = xfi(:,i,j)-squeeze(xfi(:,i,:))*Wm';
        Pf(:,:,i) = Pf(:,:,i)+Wc(j)*(diff * diff');
    end
end
end

% Analysis
function [xa,Pa] = u_mf_a(xfi,Pf,y,H,Wm,Wc,R)
Nemb = size(xfi,1);
Nx = size(xfi,2);
xa = zeros(Nemb,Nx);
Pa = zeros(Nemb,Nemb,Nx);
for i = 1:Nx
    Pyy = R;
    Pxy = zeros(Nemb);
    for j = 1:2*Nemb+1
        ydiff = xfi(:,i,j)-H(:,:,i)*squeeze(xfi(:,i,:))*Wm';
        xdiff = xfi(:,i,j)-squeeze(xfi(:,i,:))*Wm';
        Pyy = Pyy+Wc(j)*(ydiff*ydiff');
        Pxy = Pxy+Wc(j)*(xdiff*ydiff');
    end
    Kk = Pxy/Pyy; % Gain
    xa(:,i) = squeeze(xfi(:,i,:))*Wm'+Kk*(y(:,i)...
        -H(:,:,i)*squeeze(xfi(:,i,:))*Wm'); % Innovation
    Pa(:,:,i) = Pf(:,:,i)-Kk*Pyy*Kk';
end
end

%-------------------------DYNAMICS + JACOBIANS-----------------------------
%% Single-scale L96
% Model error may be introduced through perturbation of parameter a, where
% the value of 1 corresponds to the closest fit single-scale model

function dX = l96SS(t,X,a)
global Nx Fx
dX = zeros(Nx,1);
dX(1) = (X(2)-a*X(Nx-1))*X(Nx)-X(1)+Fx;
dX(2) = (X(3)-a*X(Nx))*X(1)-X(2)+Fx;
dX(Nx) = (X(1)-a*X(Nx-2))*X(Nx-1)-X(Nx)+Fx;
for i = 3:Nx-1
    dX(i) = (X(i+1)-a*X(i-2))*X(i-1)-X(i)+Fx;
end
end

function J = l96SS_TLM(t,X,a)
global Nx 
J = zeros(Nx,Nx);
J(1,:) = [-1 X(Nx) zeros(1,Nx-4) -a*X(Nx) X(2)-a*X(Nx-1)];
J(2,:) = [X(3)-a*X(Nx) -1 X(1) zeros(1,Nx-4) -a*X(1)];
J(Nx,:) = [X(Nx-1) zeros(1,Nx-4) -a*X(Nx-1) X(1)-a*X(Nx-2) -1];
for n = 3:Nx-1
    J(n,:) = [zeros(1,n-3) -a*X(n-1) X(n+1)-a*X(n-2) -1 X(n-1) zeros(1,Nx-n-1)];
end
end

%% Multi-scale L96
% Model error may be introduced through perturbation of parameter a, where
% the value of 1 corresponds to the true model

function dX = l96MS(t,X,a)
global Fx Fy Nx Ny eps hx hy
x = zeros(Nx+4,1);
x((1:Nx)+2) = X(1:Nx);
x(1:2) = X(Nx-1:Nx);
x(end-1:end) = X(1:2);
Y = X(Nx+1:Nx*(Ny+1));
Y = reshape(Y,Nx,Ny);
y = [Y(end,:); Y; Y(1,:)];
dx = zeros(Nx,1);
dy = zeros(Nx,Ny);
for n = 1:Nx
    nx = n+2;
    ny = n+1;
    dx(n) = (x(nx+1)-a*x(nx-2))*x(nx-1)-x(nx)+Fx+hx*mean(y(ny,:));
    dy(n,1) = (y(ny-1,Ny)-y(ny,3))*y(ny,2)-y(ny,1)+Fy+hy*x(nx);
    dy(n,Ny-1) = (y(ny,Ny-2)-y(ny+1,1))*y(ny,Ny)-y(ny,Ny-1)+Fy+hy*x(nx);
    dy(n,Ny) = (y(ny,Ny-1)-y(ny+1,2))*y(ny+1,1)-y(ny,Ny)+Fy+hy*x(nx);
    for j = 2:Ny-2
        dy(n,j) = (y(ny,j-1)-y(ny,j+2))*y(ny,j+1)-y(ny,j)+Fy+hy*x(nx);
    end
end
dy = dy(:)./eps;
dX = [dx; dy];
end

function J = l96MS_TLM(t,X,a)
global Nx Ny eps hx hy
    % Compute the Jacobian matrix at a given point
    x = X(1:Nx);
    y = X(Nx+1:end);
    J = zeros(Nx*(Ny+1),Nx*(Ny+1));
    J(1:Nx,1:Nx) = -eye(Nx)+circshift(eye(Nx).*circshift(x(1:Nx),1),1,2)...
        +circshift(eye(Nx).*circshift(x(1:Nx),-1),-1,2)...
        -a*circshift(eye(Nx).*circshift(x(1:Nx),2),-1,2)...
        -a*circshift(eye(Nx).*circshift(x(1:Nx),1),-2,2);
    J(1:Nx,Nx+1:end) = (hx/Ny)*repmat(eye(Nx),1,Ny);
    J(Nx+1:end,1:Nx) = (hy/eps)*repmat(eye(Nx),Ny,1);
    J(Nx+1:end,Nx+1:end) = (1/eps)*(-eye(Nx*Ny)+...
        circshift(eye(Nx*Ny).*circshift(y(:),Nx+Nx),-Nx)- ...
        circshift(eye(Nx*Ny).*circshift(y(:),Nx-2*Nx),-Nx)-...
        circshift(eye(Nx*Ny).*circshift(y(:),2*Nx-Nx),-2*Nx)+...
        circshift(eye(Nx*Ny).*circshift(y(:),-Nx-Nx),Nx));
    J((1:Nx)+Nx,(1:Nx)+2*Nx) = (1/eps)*(...
        eye(Nx).*circshift(y(end-Nx+1:end),1)-eye(Nx).*y((1:Nx)+2*Nx));
    J((1:Nx)+Nx,end-Nx+1:end) = (1/eps)*(...
        circshift(eye(Nx).*circshift(y((1:Nx)+Nx),-1),1));
    J((1:Nx)+Nx*(Ny-1),(1:Nx)+Nx) = -(1/eps)*(...
        circshift(eye(Nx).*circshift(y((1:Nx)+Nx*(Ny-1)),1),-1));
    J((1:Nx)+Nx*(Ny-1),(1:Nx)+Nx*Ny) = (1/eps)*(...
        eye(Nx).*y((1:Nx)+Nx*(Ny-3))...
        -eye(Nx).*circshift(y(1:Nx),-1)); 
    J((1:Nx)+Nx*Ny,(1:Nx)+Nx) = (1/eps)*(...
        circshift(eye(Nx).*circshift(y((1:Nx)+Nx*(Ny-2)),1),-1)...
        -circshift(eye(Nx).*y((1:Nx)+Nx),-1)); 
    J((1:Nx)+Nx*Ny,(1:Nx)+2*Nx) = -(1/eps)*(...
        circshift(eye(Nx).*y(1:Nx),-1));
    J((1:Nx)+Nx*Ny,(1:Nx)+Nx*(Ny-1)) = (1/eps)*(...
        eye(Nx).*circshift(y(1:Nx),-1));
end

%% Storm-track L96
% Time-dependent version of Bishop, Whitaker and Lei, 2017
% as in Maclean, 2025

function dX = l96ST(t, X)
   global Fx Nx
   jp1 = circshift(1:Nx, -1);
   jm1 = circshift(1:Nx, 1);
   jm2 = circshift(1:Nx, 2);

   v = 0.1; %controls the speed of storm
   list = 1:Nx;
   gamma = 0.5+2*cos((0.25*sin(v*pi*t)+list(:)/Nx)*pi).^4;
   dX = (X(jp1,:) - X(jm2,:)) .* X(jm1,:) - gamma .* X + Fx;
end 


%% Integrate dynamics
function xpost = intdyn(dynamics,xpre,T)
options = odeset('reltol',1e-4,'abstol',1e-4);
[~, xint] = ode15s(@(t,x) dynamics(t,x), [0 T], xpre, options);
xpost = xint(end,:)';
end

% Function to compute the numerical Jacobian
function TLM = embed_TLM(xa,x_embed,Nnn)
    delta = 1e-8;  % Small perturbation
    N = length(xa); % Number of variables
    TLM = zeros(N, N);

    for i = 1:N
        Xp = xa; Xm = xa;
        Xp(i) = xa(i) + delta; % Perturb state vector
        Xm(i) = xa(i) - delta;
        % Evaluate function at the perturbed state
        Fp = nearneighM(Xp,x_embed,Nnn);
        Fm = nearneighM(Xm,x_embed,Nnn);
       
        TLM(:, i) = (Fp - Fm) / (2*delta); % Approximate derivative
    end
end

% Nearest neighbour
function xpost = nearneighM(xpre,x_embed,Nnn)
xpre = xpre';
% Treat each grid point as the same, we get Nx times as much training data
% in embedded space for each grid point
Xdpre = reshape(x_embed(:,:,1:end-1),size(x_embed,1),size(x_embed,2)*(size(x_embed,3)-1))';
Xdpost = reshape(x_embed(:,:,2:end),size(x_embed,1),size(x_embed,2)*(size(x_embed,3)-1))';

d_ij = vecnorm(Xdpre-xpre,2,2);
[~,distIdxs] = sort(d_ij,'ascend'); % sort the distances
kNeigIdxs = distIdxs(1:Nnn); % indexes of the nearest neighbours
XfutNeig = Xdpost(kNeigIdxs,:); % future values of nearest neighbours
XMat = padarray(Xdpre(kNeigIdxs,:),[0 1],1,'pre'); % padded delayed neighbour vectors
alphas = (transpose(XMat)*XMat)\transpose(XMat)*squeeze(XfutNeig); % matrix form of regression
xpost = padarray(xpre,[0 1],1,'pre')*alphas;
xpost = xpost';
end


% Nudge equivalent states - currently only works for D=2, nu=2
function xnudged = nudge(x,mask_embed_equivalent,gamma)
global D nu Nx
gamma = 1-gamma;
NU = [0 -nu:-1 1:nu];
    for i = 1:Nx
        if i == 1
        x_local_eq = x;
        x_local_eq(circshift(mask_embed_equivalent==1,i-1,2)) = ...
        repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)+...
            gamma*(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent)-...
            repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1));
        x = x_local_eq;
        elseif i == 2
        x_local_eq = x;
        x_local_eq(circshift(mask_embed_equivalent==1,i-1,2)) = ...
        circshift(repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)+...
            gamma*(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent)-...
            repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)),D+1);
        x = x_local_eq;
        elseif i>2 && i<Nx-1
        x_local_eq = x;
        x_local_eq(circshift(mask_embed_equivalent==1,i-1,2)) = ...
        circshift(repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)+...
            gamma*(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent)-...
            repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)),2*(D+1));
        x = x_local_eq;
        elseif i==Nx-1
        x_local_eq = x;
        x_local_eq(circshift(mask_embed_equivalent==1,i-1,2)) = ...
        circshift(repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)+...
            gamma*(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent)-...
            repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)),3*(D+1));
        x = x_local_eq;
        elseif i==Nx
        x_local_eq = x;
        x_local_eq(circshift(mask_embed_equivalent==1,i-1,2)) = ...
        circshift(repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)+...
            gamma*(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent)-...
            repmat(mean(reshape(nonzeros(circshift(x_local_eq,-i+1,2).* ...
            mask_embed_equivalent),D+1,length(NU)),2),length(NU),1)),4*(D+1));
        x = x_local_eq;
        end
    end
    xnudged = x;
end



