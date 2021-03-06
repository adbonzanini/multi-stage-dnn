%% Initialize workspace
clear all
uqlab

% Plot settings
Fontsize = 15;
Lwidth = 2;

%% User inputs
adaptTree = 0;  % Scenario tree adaptation
wb =  0;      % Uncertainty bound
currentCEM = 0; % Current CEM
CEMtarget = 1;  % CEM target
training = 0;   % Whether to train the GP model or not (has to be trained the first time you run the code)

%% Load plasma model
sys = load('APPJmodelDAE');

% Dimensions
nx=size(sys.A,2);
nu=size(sys.B,2);
ny=size(sys.C,1);
nd=2;
nCEM = size(CEMtarget,1);


%% Define MPC Parameters

% Define cost parameters
Q = [1, 0; 0, 1];
R = 1;
PN = Q;                             % Terminal cost weight

Np = 5;                             % Prediction horizon
N = 20;                             % Simulation horizon


% Initial point
yi = [4;0];


% Uncertainty bounds
wl = -1*[wb;0];
wu =  1*[wb;0];


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

    % Remove in the final version !!
    %trainSplit = 1:2:200;
    %Xtrain = Xtrain(trainSplit,:);
    %Ytrain = Ytrain(trainSplit,:);

    myKrigingMat = trainGP(Xtrain, Ytrain, Xtest, Ytest, 0);
    
    % Create GP training object
    GPtraining.myKrigingMat = myKrigingMat;
    GPtraining.Xtrain = Xtrain;
    GPtraining.Ytrain = Ytrain;
    GPtraining.Xtest = Xtest;
    GPtraining.Ytest = Ytest;
    GPtraining.lag = lag;
else
    load('GPtraining.mat')
end




%%
% msMPC
[yTr,uOptSeq, ssPlot, CEMplot, Y, U] = msMPC(yi, sys, currentCEM, CEMtarget, wl, wu, Np, N, Q, R, PN, GPtraining, adaptTree);

% Plot figures

figure(1)
hold on
plot([0:N], CEMplot, 'LineWidth', Lwidth);
plot([0, N], [CEMtarget, CEMtarget], 'k--', 'LineWidth', Lwidth);
ylim([0, CEMtarget*1.1])
xlabel('Time Step (k)')
ylabel('CEM')
set(gca,'FontSize',Fontsize)
box on


figure(2)
subplot(2,1,1)
hold on
plot([0:N], yTr(1,:), 'LineWidth', Lwidth)
% plot([0:N], ssPlot(1,1:end-1), 'k-', 'LineWidth', Lwidth)
plot([0, N], max(Y.V(:,1))*ones(1,2), 'k--', 'LineWidth', Lwidth)
plot([0, N], min(Y.V(:,1))*ones(1,2), 'k--', 'LineWidth', Lwidth)
ylim([min(Y.V(:,1))-1, max(Y.V(:,1))+2])
ylabel('y_1')
set(gca,'FontSize',Fontsize)
box on
subplot(2,1,2)
hold on
plot([0:N], yTr(2,:), 'LineWidth', Lwidth)
% plot([0:N], ssPlot(2,1:end-1), 'k-', 'LineWidth', Lwidth)
plot([0, N], max(Y.V(:,2))*ones(1,2), 'k--', 'LineWidth', Lwidth)
plot([0, N], min(Y.V(:,2))*ones(1,2), 'k--', 'LineWidth', Lwidth)
ylim([min(Y.V(:,2))-1, max(Y.V(:,2))+2])
xlabel('Time Step (k)')
ylabel('y_2')
set(gca,'FontSize',Fontsize)
box on


figure(3)
subplot(2,1,1)
hold on
plot([0:N-1], uOptSeq(1,:), 'LineWidth', Lwidth)
plot([0, N], max(U.V(:,1))*ones(1,2), 'k--', 'LineWidth', Lwidth)
plot([0, N], min(U.V(:,1))*ones(1,2), 'k--', 'LineWidth', Lwidth)
ylim([min(U.V(:,1))-0.5, max(U.V(:,1))+0.5])
ylabel('q')
set(gca,'FontSize',Fontsize)
box on
subplot(2,1,2)
hold on
plot([0:N-1], uOptSeq(2,:), 'LineWidth', Lwidth)
plot([0, N], max(U.V(:,2))*ones(1,2), 'k--', 'LineWidth', Lwidth)
plot([0, N], min(U.V(:,2))*ones(1,2), 'k--', 'LineWidth', Lwidth)
ylim([min(U.V(:,2))-0.5, max(U.V(:,2))+0.5])
xlabel('Time Step (k)')
ylabel('P')
set(gca,'FontSize',Fontsize)
box on

uOptSeq

%{
figure(3)
hold on
plot(X, 'color', [1, 1, 1]*0.9, 'LineWidth', Lwidth, 'LineStyle', '--')
plot(Cinf, 'color', [1, 1, 1]*0.8)
plot(Cinf_ob, 'color', [1, 1, 1]*0.7)
plot(yTr(1,:), yTr(2,:), '.-','Markersize', 20, 'LineWidth', Lwidth)
plot(ysp1(1), ysp1(2), 'kx', 'Markersize', 10)
plot(ysp2(1), ysp2(2), 'kx', 'Markersize', 10)


if saveSwitch==1
    if gpSwitch==1 && worstCase==0
        saveStr = 'SD_';
    elseif gpSwitch==1 && worstCase==1
        saveStr = 'WC_';
    end
    
    if useProj==0
        saveStr2 = 'NoProj_';
    else
       saveStr2 = 'YesProj_'; 
    end
    
    save(['../Output-Data-Files/LB-MS-MPC_', saveStr, saveStr2, datestr(now,'YYYY-mm-dd_HH_MM_SS'), ], 'X', 'Cinf', 'Cinf_ob', 'yTr')
end
%}