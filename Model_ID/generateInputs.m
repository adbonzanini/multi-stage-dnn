clear all

% Timestamp and directory for saving
timeStamp = datestr(now,'yyyymmdd_HHMMSS');
directory = '/users/adbonzanini/Box Sync/Berkeley/Research/Explicit MPC Paper/DNN-MPC-Plasma/model_ID/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFINE SOME VALUES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Overall time period
Toverall = 1000;
Tsampling = 1.3;
disp(['Experiment Time = ', num2str(Toverall*0.8/60), ' min']) % in minutes

%Initial conditions (if we want to start it somewhere in particular)
wi = 3;
qi = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PICK COMBINATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Values that we want to examine

% Relatively small changes (likely to be in linear region)
w = [wi, 1.5, 2.0, 3]; % 1.5-5 W
q = [qi, 1.5, 2.0, 4]; % 0.5-5 slm 

% Changes farther away from equilibrium (likely to be in the nonlinear region)
wFar = [3.5];%, 4 5];
qFar = [3.5];%, 0.8, 1];

%Take all possible combinations
u = combvec(w, q);

uFar = combvec(wFar, qFar);

p = [unifrnd(-0.15, 0.15, 1, size(u,2));unifrnd(-0.15, 0.15, 1, size(u,2))];

% How many steps given these choices?
N = size(u,2);
Ntot = size([u, uFar],2);

% How long is each change held at that point?
T = ceil(Toverall/Ntot);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHUFFLE THE SEQUENCE OF COMBINATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define a matrix containing 1s and 0s that will shuffle the permutations around
orderMat = zeros(N, N);
orderMat(1,1) = 1;         %We don't want to shuffle our initial conditions


rng(1) %For repeatability
r = 1+randperm(N-1);
for j = 2:N
    orderMat(j, r(j-1)) = 1;
end


%Input sequence
U = u*orderMat;
% Umod = U +p;

U = [U, uFar];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT STEP FIGURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
timeAxis = [1:T:Toverall+T];

% Discard some combinations
U(:,find(U(1,:)>=4 & U(2,:)<=1)) =[];
U = U(:,1:end);
% [~, idx] = sort(U-[wi;qi],2, 'ComparisonMethod','abs');
% U(1,:) = U(1,idx(1,:));
% U(2,:) = U(2,idx(2,:));

timeAxis = timeAxis(:,1:size(U,2)+1);

% How many steps given these choices?
N = size(U,2);

% How long is each change held at that point?
T = ceil(Toverall/N);
disp(['Time steps per change = ', num2str(T), ' steps'])


figure(1)
subplot(2, 1, 1)
stairs(timeAxis, [U(1, 1), U(1,:)])
xlabel('Time/s')
ylabel('Power')
ylim([0.5, 5.5])
subplot(2, 1, 2)
stairs(timeAxis, [U(2,2), U(2,:)])
xlabel('Time/s')
ylabel('Flow (q)/slm')
ylim([0.5, 5.5])

figure(2)
subplot(1, 2, 1)
stairs(U(1,:), U(2,:), 'o')
xlabel('Power')
ylabel('Flow')
subplot(1, 2, 2)
stairs(U(1,:), U(2,:), 'o')
xlabel('Power')
ylabel('Flow')

save([directory, timeStamp, 'inputTrajectories.mat'], 'U', 'T')
