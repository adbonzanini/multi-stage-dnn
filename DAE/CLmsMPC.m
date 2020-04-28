%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script runs closed-loop simulations of a multi-stage MPC with
% scenario tree adaptation
%
% Set user inputs under "User inputs" line
%
% Written by: Angelo D. Bonzanini
% Last edited: April 27 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set up workspace
% clear all
addpath('Model_ID')
uqlab
Fontsize = 15;
Lwidth = 2;
color = {[0.6350, 0.0780, 0.1840], [0 0.4470 0.7410], [0.4940, 0.1840, 0.5560], [0.8500, 0.3250, 0.0980]}; %red, blue, purple, orange 


%% Load plasma model and training data
sys = load('Model_ID/APPJmodelDAE');
load('Supporting-Data-Files/DNN_training.mat')

%% User inputs
caseidx = 3;    %1 = nominal; 2 = multi-stage; 3 = adaptive multi-stage;
sdNoise = 0.4;  % standard deviation of the random  noise (NOT due to mismatch)
CEMtarget = 10;  % CEM target
currentCEM = 0; % Initial CEM
training = 0;   % Whether to train the GP model or not (has to be trained the first time you run the code)
Kcem = 0.5; 

if caseidx==1
    wb = 0; adaptTree=0;
elseif caseidx==2
    wb = sys.maxErrors(1)+2*sdNoise; adaptTree=0;
elseif caseidx==3
    wb = sys.maxErrors(1)+2*sdNoise; adaptTree=1;
else
    error('Invalid case study')
end

% Dimensions
nx=size(sys.A,2);
nu=size(sys.B,2);
ny=size(sys.C,1);
nd=2;

nCEM = size(CEMtarget,1);



%% Learn Guassian process (GP) model
if training == 1
    % Extract data from model identification
    xdata = sys.xdata;
    udata = sys.udata;
    xCompare = sys.xCompare;

    % Generate training and test data
    lag = 3;
    XX = [xdata(1:end-lag,:), udata(1:end-lag, :)];
    YY = xCompare(lag+1:end, :) - xdata(lag+1:end, :);
    % Stack rows side by side to account for lag
    Nsamp = size(XX,1);

    Xtrain = zeros(Nsamp, (ny+nu)*lag);
    Ytrain = YY;

    zeros(Nsamp, ny);
    for j = 1:Nsamp-lag
        Xrow = XX(j:j+lag-1,:)';
        Xtrain(j,:) = Xrow(:)';
    end

    % Initialize vectors to store the test data after each OCP solution
    Xtest = Xtrain(1:400,:);
    Xtrain = Xtrain(400:end, :);
    Ytest = Ytrain(1:400,:);
    Ytrain = Ytrain(400:end, :);
    
    showTrainingPlots = 1;
    myKrigingMat = trainGP(Xtrain, Ytrain, Xtest, Ytest, showTrainingPlots);
    
    % Reset test inputs and outputs so that they run on the first step of the OCP
    Xtest = zeros(1, size(Xtrain,2));
    Ytest = zeros(1, size(Ytrain,2));
    
    % Create GP training object
    GPtraining.myKrigingMat = myKrigingMat;
    GPtraining.Xtrain = Xtrain;
    GPtraining.Ytrain = Ytrain;
    GPtraining.Xtest = Xtest;
    GPtraining.Ytest = Ytest;
    GPtraining.lag = lag;
    save('GPtraining.mat', 'GPtraining')
else
    load('GPtraining.mat')
    myKrigingMat = GPtraining.myKrigingMat;
    Xtrain = GPtraining.Xtrain;
    Ytrain = GPtraining.Ytrain;
    Xtest = GPtraining.Xtest;
    Ytest=GPtraining.Ytest;
    lag = GPtraining.lag;
end



%% Define MPC Parameters
Np = 6;      % Prediction horizon (ensure it is the same as DNN)
Nsim = 60;   % Simulation horizon

% Initial point
yi = [-2;0];

% Uncertainty bounds
wl = -1*[wb;0];
wu =  1*[wb;0];


%% Setup the mpc problem
[solver, args, Y, U] = msMPCsolver(yi, sys, currentCEM, CEMtarget, wl, wu, Np, eye(nx), eye(nu), eye(nx), GPtraining, Kcem);


%% Initialize empty vectors
CEMplot = zeros(1, Nsim+1);
yplot = zeros(ny,Nsim+1);
uplot = zeros(nu,Nsim);
GPstore = zeros(ny, Nsim+1);
sdGPstore = zeros(ny, Nsim+1);

% Calculate initial steady-state
[xd0, xa0, d0, uss] = DAEssCalc(yi(1)+sys.steadyStates(1),4, 0);
xki = [xd0(2)*300-273;xa0(1)]-sys.steadyStates(1:2)';
yplot(:,1) = xki;
dataIn = [yplot(:,1)', CEMplot(1), wb, 0];

% Define noise
rng(3)
wReal = [normrnd(0, sdNoise, [1, Nsim]); normrnd(0, 0.2, [1, Nsim])];

%% MPC Loop
for k=1:Nsim

    [U_mpc, Feas, V_opt] = solveSamplesMPC(solver, args, dataIn);
    
    xPred = A*xki+B*U_mpc';
    
    Fsim = plantSimulator(xd0, U_mpc'+sys.steadyStates(4:5)', d0, xa0);
    xd0 = full(Fsim.xf);
    xa0 = full(Fsim.zf);
   

    xki = [xd0(2)*300-273;xa0(1)]-sys.steadyStates(1:2)'+wReal(:,k);
    %xki = A*xki+B*U_mpc';
    yki = xki;
    yplot(:, k+1) = yki;
    
    
    
%     if yki(1)+sys.steadyStates(1)<=43
%         addCEM = Kcem^(43-(yki(1)+sys.steadyStates(1)));
%     elseif yki(1)+sys.steadyStates(1)>=43
%         addCEM = 2*Kcem^(43-(yki(1)+sys.steadyStates(1)));
%     end
    addCEM = Kcem^(43-(yki(1)+sys.steadyStates(1)));
    CEMplot(k+1) = CEMplot(k)+ addCEM;
    currentCEM = CEMplot(k+1);
    
    
    % Update test dataset
    Xtest(1,1:end-ny) = Xtest(1,ny+1:end);
    Xtest(1,end-ny+1:end) = (xPred - xki)';
    Xtest(1,(lag-1)*nu+1:lag*nu) = U_mpc(:,1)';
    [GP,varGP] = uq_evalModel(myKrigingMat,Xtest);
    
    sdGPstore(:,k) = sqrt(varGP);
    GPstore(:,k) = abs(GP(1));
    % Update wb through GP
    if adaptTree == 1
        wb(1) = abs(GP(1))+2*sdNoise;
        
    end
    
    
    
    dataIn = [yki',currentCEM, wb, 0];
    
    
    
end



%% Plot figures

% Figure position and size
figpos = [300, 300, 750, 750];

% Change back to non-deviation variables
yplot = yplot+sys.steadyStates(1:2)';
%%
figure(1)
set(gcf, 'Position', figpos)
subplot(3,1,1)
hold on
h{caseidx} = plot(0:Nsim, CEMplot, '-o', 'color', color{caseidx}, 'Linewidth', Lwidth);
plot([0, Nsim], [CEMtarget, CEMtarget], 'k-', 'Linewidth', Lwidth)
xlabel('Time (s)')
ylabel('CEM (min)')
ylim([0, CEMtarget+0.5])
set(gca, 'Fontsize', Fontsize)
box on
%
subplot(3,1,2)
hold on
plot(0:Nsim, yplot(1,:), '-o', 'color', color{caseidx}, 'Linewidth', Lwidth)
plot([0, Nsim], [max(Y.V(:,1)), max(Y.V(:,1))]+sys.steadyStates(1), 'k--', 'Linewidth', Lwidth)
plot([0, Nsim], [min(Y.V(:,1)), min(Y.V(:,1))]+sys.steadyStates(1), 'k--', 'Linewidth', Lwidth)
ylim([min(Y.V(:,1))-0.5, max(Y.V(:,1))+2.5]+sys.steadyStates(1))
xlabel('Time (s)')
ylabel('T (^{\circ}C)')
set(gca, 'Fontsize', Fontsize)
box on
%
subplot(3,1,3)
hold on
plot(0:Nsim, yplot(2,:), '-o', 'color', color{caseidx}, 'Linewidth', Lwidth)
plot([0, Nsim], [max(Y.V(:,2)), max(Y.V(:,2))]+sys.steadyStates(2), 'k--', 'Linewidth', Lwidth)
plot([0, Nsim], [min(Y.V(:,2)), min(Y.V(:,2))]+sys.steadyStates(2), 'k--', 'Linewidth', Lwidth)
ylim([min(Y.V(:,2))-0.5, max(Y.V(:,2))+0.5]+sys.steadyStates(2))
xlabel('Time (s)')
ylabel('I (mA)')
set(gca, 'Fontsize', Fontsize)

subplot(3,1,1)

%{
legend([h{1}, h{2}, h{3}], 'nominal NMPC', 'msNMPC', 'Adaptive msNMPC')
title('Np = 6')
%}

disp('Average GP prediction')
disp(mean(GPstore,2))


