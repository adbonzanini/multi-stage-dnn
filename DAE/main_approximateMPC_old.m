%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script runs closed-loop simulations comparing the performance of the
% NMPC vs the DNN-based approximate NPMC. Requires to have trained the DNN
% first by running the main_approximate_dnn.m file.
%
% Set user inputs under "User inputs" line
%
% Written by: Angelo D. Bonzanini
% Last edited: April 26 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set up workspace
% clear all
uqlab
Fontsize = 15;
Lwidth = 2;
color = {[0.6350, 0.0780, 0.1840], [0 0.4470 0.7410], [0.4940, 0.1840, 0.5560], [0.8500, 0.3250, 0.0980]}; %red, blue, purple, orange 
warning('on','all')
addpath('Model_ID')

%% User inputs
approximate=0;  % Use DNN or not
wb =  5;        % (Initial) uncertainty bound
adaptTree = 0;  % Scenario tree adaptation
currentCEM = 0; % Initial CEM
CEMtarget = 1;  % CEM target
training = 0;   % Whether to train the GP model or not (has to be trained the first time you run the code)


%% Load plasma model and training data
sys = load('Model_ID/APPJmodelDAE');
DNNstruct = load('Supporting-Data-Files/DNN_training.mat');
load('Supporting-Data-Files/DNN_training.mat')
% Run casadi function for faster DNN evaluation
c = casadiDNN(DNNstruct, 6, DNNstruct.L);
dnnmpcFn = c.dnnmpc;

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
    lag = 2;
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
    Xtest = zeros(1, size(Xtrain,2));
    Ytest = zeros(1, size(Ytrain,2));
    
%     % for testing
%     Xtest = Xtrain(end*0.8:end,:);
%     Ytest = Ytrain(end*0.8:end,:);
%     Xtrain = Xtrain(1:end*0.8,:);
%     Ytrain = Ytrain(1:end*0.8,:);

    [gprMdl1, gprMdl2, kfcn] = trainGP(Xtrain, Ytrain, Xtest, Ytest, 0);
    
    % Create GP training object
    GPtraining.gprMdl1 = gprMdl1;
    GPtraining.gprMdl2 = gprMdl2;
    GPtraining.kfcn = kfcn;
    GPtraining.Xtrain = Xtrain;
    GPtraining.Ytrain = Ytrain;
    GPtraining.Xtest = Xtest;
    GPtraining.Ytest = Ytest;
    GPtraining.lag = lag;
    save('GPtraining.mat', 'GPtraining')
    warning('GP training overwritten!')
else
    load('GPtraining.mat')
    gprMdl1 = GPtraining.gprMdl1;
    gprMdl2 = GPtraining.gprMdl2;
    kfcn = GPtraining.kfcn;
    Xtrain = GPtraining.Xtrain;
    Ytrain = GPtraining.Ytrain;
    Xtest = GPtraining.Xtest;
    Ytest=GPtraining.Ytest;
    lag = GPtraining.lag;
end


%% Define MPC Parameters
Np = 5;      % Prediction horizon (ensure it is the same as DNN)
Nsim = 20;   % Simulation horizon
Kcem = 0.75; 

% Initial point
yi = [4;0];

% Uncertainty bounds
wl = -1*[wb;0];
wu =  1*[wb;0];


%% Setup the mpc problem
[solver, args, Y, U] = msMPCsolver(yi, sys, currentCEM, CEMtarget, wl, wu, Np, eye(nx), eye(nu), eye(nx), GPtraining, Kcem, GPinPredictionIdx);


%% Initialize empty vectors
CEMplot = zeros(1, Nsim+1);
yplot = zeros(ny,Nsim+1);
uplot = zeros(nu,Nsim);
uopt = zeros(nu, Np);

% Calculate initial steady-state
[xd0, xa0, d0, uss] = DAEssCalc(yi(1)+sys.steadyStates(1),4, 0);
xki = [xd0(2)*300-273;xd0(1)*300-273]-sys.steadyStates(1:2)';
yplot(:,1) = xki;
dataIn = [yplot(:,1)', CEMplot(1), wb, 0];

%% MPC Loop
for k=1:Nsim
    
    if approximate ==0
    [U_mpc, Feas, V_opt] = solveSamplesMPC(solver, args, dataIn);
    else
        U_mpc = full(dnnmpcFn(dataIn'))';
    end
    
    xPred = A*xki+B*U_mpc';
    
    Fsim = plantSimulator(xd0, U_mpc'+sys.steadyStates(3:4)', d0, xa0);
    xd0 = full(Fsim.xf);
    xa0 = full(Fsim.zf);
   
    xki = [xd0(2)*300-273;xa0(1)]-sys.steadyStates(1:2)'+[3;0];
    %xki = A*xki+B*U_mpc';
    yki = xki;
    yplot(:, k+1) = yki;
    
    
    addCEM = Kcem^(43-(yki(1)+sys.steadyStates(1)));
    CEMplot(k+1) = CEMplot(k)+ addCEM;
    currentCEM = CEMplot(k+1);
    
    
    % Update test dataset
    Xtest(1,1:end-ny) = Xtest(1,ny+1:end);
    Xtest(1,end-ny+1:end) = (xPred - xki)';
    Xtest(1,(lag-1)*nu+1:lag*nu) = uopt(:,1)';
%     [GP,~] = uq_evalModel(myKrigingMat,Xtest);
    [GP, ~] = predict(gprMdl1, Xtest);
    
    % Update wb through GP
    if adaptTree == 1
        wb(1) = abs(GP(1)');
    end
    
    
    
    dataIn = [yki',currentCEM, wb, 0];
    
    
    
end




%% Plot figures

figpos = [300, 300, 750, 750];
figure(1)
set(gcf, 'Position', figpos)
subplot(3,1,1)
hold on
h{approximate+adaptTree+1} = plot(0:Nsim, CEMplot, '-o', 'color', color{approximate+adaptTree+1}, 'Linewidth', Lwidth);
plot([0, Nsim], [CEMtarget, CEMtarget], 'k-', 'Linewidth', Lwidth)
xlabel('Time (s)')
ylabel('CEM (min)')
ylim([0, CEMtarget+0.1])
set(gca, 'Fontsize', Fontsize)
%legend([h{}], 'NMPC')
box on
%
subplot(3,1,2)
hold on
plot(0:Nsim, yplot(1,:), '-o', 'color', color{approximate+adaptTree+1}, 'Linewidth', Lwidth)
plot([0, Nsim], [max(Y.V(:,1)), max(Y.V(:,1))], 'k--', 'Linewidth', Lwidth)
plot([0, Nsim], [min(Y.V(:,1)), min(Y.V(:,1))], 'k--', 'Linewidth', Lwidth)
ylim([min(Y.V(:,1))-0.5, max(Y.V(:,1))+2.5])
xlabel('Time (s)')
ylabel('T (^{\circ}C)')
set(gca, 'Fontsize', Fontsize)
box on
%
subplot(3,1,3)
hold on
plot(0:Nsim, yplot(2,:), '-o', 'color', color{approximate+adaptTree+1}, 'Linewidth', Lwidth)
plot([0, Nsim], [max(Y.V(:,2)), max(Y.V(:,2))], 'k--', 'Linewidth', Lwidth)
plot([0, Nsim], [min(Y.V(:,2)), min(Y.V(:,2))], 'k--', 'Linewidth', Lwidth)
ylim([min(Y.V(:,2))-0.5, max(Y.V(:,2))+0.5])
xlabel('Time (s)')
ylabel('y_2')
set(gca, 'Fontsize', Fontsize)

subplot(3,1,1)

%{
legend([h{1}, h{2}], 'NMPC', 'Approximate NMPC')
legend([h{1}, h{2}, h{3}], 'NMPC', 'Approximate NMPC', 'Approximate NMPC with GP adaptation')
title('w_b = 5')
title('w_b = 5')
%}




