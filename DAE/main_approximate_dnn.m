%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script generates data by solving the OCP for different initial
% states and parameters (i.e., curent CEM values and uncertainty bounds)
% to obtain the optimal inputs to the system. Then, it trains a DNN that
% takes as inputs the initial states and parameters and yields as outputs
% the optimal inputs to the system. 
%
% That is, [x1, x2, p1, p2] --> DNN --> [u1, u2]
%
% User inputs can be found under $%% User inputs
%
% Written by: Angelo D. Bonzanini
% Last edited: April 24 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Clear variables
%% Set up workspace
yalmip('clear')
% Start uqlab
uqlab
% Fix random seed
rng(100, 'twister')
% Plot settings
Fontsize = 15;
Lwidth = 2;

%% User inputs
CEMtarget = 1;  % CEM target
training = 0;   % Whether to train the GP model or not (Not needed for DNN training, only for MPC)
currentCEM = 0; % Current CEM (initial) -- leave at zero


%% Parameter spaces from which to draw points for training
% CEM space from which to draw points
CEM_min = 0;
CEM_max = 1.5;

% Uncertainty bounds -- these defines the space from which to draw points [-wb, wb]
wb1 =  5;      
wb2 =  0.1;
wl = -1*[wb1;wb2];
wu =  1*[wb1;wb2];



%% Set up MPC problem
directory = pwd;
sys = load('APPJmodelDAE');

A = sys.A; B=sys.B; C=sys.C; D=0;
Tss = sys.steadyStates(1);Iss = sys.steadyStates(2);
Pss = sys.steadyStates(4); qss = sys.steadyStates(5);

% Dimensions
nx=size(sys.A,2);
nu=size(sys.B,2);
ny=size(sys.C,1);
nw=2;
nCEM = size(CEMtarget,1);

% Define cost parameters
Q = [1, 0; 0, 1];
R = 1;
PN = Q;

% MPC parameters
Np = 5;      % Prediction horizon
N = 1;       % Simulation horizon

% Initial point
yi = [4;0];



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
    save('GPtraining', 'GPtraining');
else
    GPtraining = load('GPtraining').GPtraining;
end


%%
% Setup the mpc problem
[solver, args, Y, U] = msMPCsolver(yi, sys, currentCEM, CEMtarget, wl, wu, Np, Q, R, PN, GPtraining);


% Output space from which to draw points
y_min = [min(Y.V(:,1)), min(Y.V(:,2))];
y_max = [max(Y.V(:,1)), max(Y.V(:,2))];

% Input space
u_min = [min(U.V(:,1)), min(U.V(:,2))];
u_max = [max(U.V(:,1)), max(U.V(:,2))];

%%
% Calculate feasible region of mpc problem
DoA = Y;
for i = 1:Np
    pre = inv(A) * (DoA + (-B*U));
    DoA1 = pre & Y;
    if DoA1 == DoA
        i;
        break    
    else
        DoA = DoA1;
    end
end


%% Create input objects


for i = 1:nx
    Input.Marginals(i).Type = 'Uniform';
    Input.Marginals(i).Parameters = [y_min(i), y_max(i)];    
end

% Create state "input" object
myInput_X = uq_createInput(Input);

% Marginals for the reference
for i = 1:nCEM
    Input.Marginals(nx+i).Type = 'Uniform';
    Input.Marginals(nx+i).Parameters = [CEM_min(i), CEM_max(i)];
end


for i = 1:nw
    Input.Marginals(nx+nCEM+i).Type = 'Uniform';
    Input.Marginals(nx+nCEM+i).Parameters = [wl(i), wu(i)];
end

% Create total "input" object (both states and reference)
myInput_P = uq_createInput(Input);



%% Sample within the domain of attraction

% Specify number of samples
Nsamp = 10000;

% Sample the state/reference space
Psamp = uq_getSample(myInput_P, 10*Nsamp, 'MC');

% Check which points are inside DoA
index = DoA.contains(Psamp(:,1:nx)');
data_rand = Psamp(index,:);
data_rand = data_rand(1:Nsamp,:);


%%
[U_mpc, Feas, V_opt] = solveSamplesMPC(solver, args, data_rand);

target_rand = [U_mpc];

% Combine data
data = [data_rand];
target = [target_rand];

%{
%% Test a sample mpc
Nsim = 20;
wb=0;
CEMplot = zeros(1, Nsim);
yi = [4;0];
yplot = zeros(ny,Nsim);

Kcem = 0.75;

[xd0, xa0, d0, uss] = DAEssCalc(yi(1)+sys.steadyStates(1),4, 0);
xki = [xd0(2)*300-273;xa0(1)]-sys.steadyStates(1:2)';
yplot(:,1) = xki;
data_rand = [yplot(:,1)', CEMplot(1), wb, 0];


for k=1:Nsim
    [U_mpc, Feas, V_opt] = solveSamplesMPC(solver, args, data_rand);
    
    Fsim = plantSimulator(xd0, U_mpc'+sys.steadyStates(4:5)', d0, xa0);
    xd0 = full(Fsim.xf);
    xa0 = full(Fsim.zf);
   
    xki = [xd0(2)*300-273;xa0(1)]-sys.steadyStates(1:2)'+[3;0];
    %xki = A*xki+B*U_mpc';
    yki = xki;
    yplot(:, k+1) = yki;
    
    
    addCEM = Kcem^(43-(yki(1)+sys.steadyStates(1)));
    CEMplot(k+1) = CEMplot(k)+ addCEM;
    currentCEM = CEMplot(k+1);
    
    data_rand = [yki',currentCEM, wb, 0];
end

figure(1)
hold on
plot([0:Nsim], CEMplot, '--', 'Linewidth', 2)
ylim([0, 1.1])
figure()

figure(2)
subplot(2,1,1)
hold on
plot([0:Nsim], yplot(1,:), '--')
plot([0, Nsim], max(Y.V(:,1))*ones(1,2), 'k--', 'LineWidth', Lwidth)
plot([0, Nsim], min(Y.V(:,1))*ones(1,2), 'k--', 'LineWidth', Lwidth)
ylim([min(Y.V(:,1))-1, max(Y.V(:,1))+2])
%%
%}

%%
% Scale variables
xscale_min = [y_min, CEM_min, wl(1), wl(2)];
xscale_max = [y_max, CEM_max, wu(1), wu(2)];
x = (data - repmat(xscale_min,[size(data,1),1]))./(repmat(xscale_max-xscale_min,[size(data,1),1]));
x = x';
tscale_min = u_min;
tscale_max = u_max;
t = (target - repmat(tscale_min,[size(data,1),1]))./(repmat(tscale_max-tscale_min,[size(data,1),1]));
t = t';

% List of nodes and layers
Nlayers_list = 5; %2:1:10; 
Nnodes_list = 6; %2:2:10; %6
mse_tol = 1e-5;

% Fit deep neural network for each hyperparameter
net_list = cell(length(Nlayers_list),length(Nnodes_list));
mse_list = zeros(length(Nlayers_list), length(Nnodes_list));
Memory_dnn_kb = zeros(length(Nlayers_list), length(Nnodes_list));
index = 0;
%%
for i = 1:length(Nlayers_list)
    for j = 1:length(Nnodes_list)
        fprintf('\n Training %d of %d ...', [length(Nnodes_list)*(i-1)+j, length(Nlayers_list)*length(Nnodes_list)])
        index = index + 1;
        net = feedforwardnet(Nnodes_list(j)*ones(1,Nlayers_list(i)), 'trainlm');
        for l = 1:Nlayers_list(i)
            net.layers{l}.transferFcn = 'poslin';
        end
        [net,tr] = train(net, x, t);
        net_list{i,j} = net;
        mse_list(i,j) = tr.best_perf;
        
        % Calculate memory
        M = Nnodes_list(j);
        L = Nlayers_list(i);
        ninput = nx+ny;
        noutput = nu;
        Mdnn = (ninput+1)*M + (L-1)*(M+1)*M + (M+1)*noutput;
        Memory_dnn_kb(i,j) = Mdnn*8/1e3;   
        fprintf('Done!\n')
    end
end



%%


% Save variables used later so that we can run the DNN-controller in a separate file
save('Supporting-Data-Files/DNN_training.mat','net','xscale_min','xscale_max', 'tscale_min', 'tscale_max', ...
    'y_min', 'y_max', 'u_min', 'u_max', 'A', 'B', 'C', 'nx', 'nu', 'ny', 'nCEM', 'Y', 'U', 'Q', 'R', 'PN', 'mse_list', ...
    'currentCEM', 'CEM_min', 'CEM_max', 'CEMtarget', 'N', 'data_rand', 'target_rand','L')

if length(Nlayers_list)>1
    fprintf('\n')
    warning('MSE File Overwritten!')
    save(['Supporting-Data-Files/MSE_Ns_', num2str(Nsamp), '.mat'], 'Nlayers_list', 'Nnodes_list', 'mse_list', 'Memory_dnn_kb')
end
%}

