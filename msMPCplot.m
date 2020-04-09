%% Initialize workspace
clear all

% Plot settings
Fontsize = 15;
Lwidth = 2;


%% MPC Parameters

% Define cost parameters
Q = [2, 0; 0, 1];
R = 1;
PN = Q;                            % Terminal cost weight

Np = 4;                             % Prediction horizon
N = 20;                             % Simulation horizon


% Initial point
yi = [2;0];

% Uncertainty realizations
wl = -[1.6;0];
wu =  [1.6;0];


[yTr,uOptSeq, ssPlot, Y, U] = msMPC(yi, wl, wu, Np, N, Q, R, PN);


figure(1)
subplot(2,1,1)
hold on
plot([0:N], yTr(1,:), 'LineWidth', Lwidth)
plot([0:N], ssPlot(1,1:end-1), 'k-', 'LineWidth', Lwidth)
ylim([min(Y.V(:,1)), max(Y.V(:,1))+2])
ylabel('y_1')
set(gca,'FontSize',Fontsize)
box on
subplot(2,1,2)
hold on
plot([0:N], yTr(2,:), 'LineWidth', Lwidth)
plot([0:N], ssPlot(2,1:end-1), 'k--', 'LineWidth', Lwidth)
ylim([min(Y.V(:,2)), max(Y.V(:,2))])
xlabel('Time Step (k)')
ylabel('y_2')
set(gca,'FontSize',Fontsize)
box on


figure(2)
subplot(2,1,1)
hold on
plot([0:N-1], uOptSeq(1,:), 'LineWidth', Lwidth)
ylim([min(U.V(:,1)), max(U.V(:,1))])
ylabel('q')
set(gca,'FontSize',Fontsize)
box on
subplot(2,1,2)
hold on
plot([0:N-1], uOptSeq(2,:), 'LineWidth', Lwidth)
ylim([min(U.V(:,2)), max(U.V(:,2))])
xlabel('Time Step (k)')
ylabel('P')
set(gca,'FontSize',Fontsize)
box on



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








