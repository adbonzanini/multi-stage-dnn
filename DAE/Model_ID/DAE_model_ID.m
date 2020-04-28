clear all

% Get current directory
directory = pwd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD COLLECTED DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose if you want to overwrite the datasets
overwriteData = 1;

% Sampling time
Tsampling = 0.5;

% Load DAE data
data = load('DAE_data');

x1_data = data.xdVec(2,:)*300-273; % Ts in oC
x2_data = data.xaVec(1,:);         % ip in mA
x3_data = data.xaVec(3,:);         % Rp in 1e-5 Ohm
u1_data = data.uSequence(1,:);     % P_set in W
u2_data = data.uSequence(2,:);     % Flow in slm

% Plot the data to visualize it
figure(1)
subplot(5, 1, 1)
plot(x1_data)
ylabel('T_s ( ^\circC)')
subplot(5, 1, 2)
plot(x2_data)
ylabel('i_p (mA)')
subplot(5, 1, 3)
plot(x3_data)
ylabel('R_p (\mu\Omega)')
subplot(5, 1, 4)
stairs(u1_data)
ylabel('Power (W)')
subplot(5, 1, 5)
stairs(u2_data)
ylabel('Flow (slm)')
xlabel('Time Instance')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IDENTIFY THE MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Translate to the origin
x1ss = mean(x1_data(1:5));     % Ts
x2ss = mean(x2_data(1:5));     % ip
x3ss = mean(x3_data(1:5));     % Rp
u1ss = mean(u1_data(1:5));     % P
u2ss = mean(u2_data(1:5));     % q

steadyStates = [x1ss, x2ss, x3ss, u1ss, u2ss];
disp(['Steady States = ', num2str(steadyStates)]);


% Work with deviation variables
x1data = x1_data -x1ss;
x2data = x2_data -x2ss;
x3data = x3_data -x3ss;
u1data = u1_data -u1ss;
u2data = u2_data -u2ss;


xdata = [x1data; x2data]';
udata = [u1data; u2data]';

subIDdata = iddata(xdata, udata, Tsampling);
Ndata = subIDdata.N;

modelOrder = size(xdata,2);
% opt = ssestOptions('OutputWeight', [1,0;0,1]);
% sys = ssest(subIDdata,modelOrder, 'DisturbanceModel', 'none', 'Form', 'canonical', 'Ts',Tsampling);
sys = n4sid(subIDdata, modelOrder, 'DisturbanceModel', 'none', 'Form', 'canonical', 'Ts',Tsampling);
% sys = ssregest(subIDdata,modelOrder, 'DisturbanceModel', 'none','Form', 'canonical', 'Ts',1.3);

A = sys.A;
B = sys.B; 
C = sys.C; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERIFY THE MODEL GRAPHICALLY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

simTime = 0:Tsampling:Tsampling*(size(udata,1)-1);
xCompare = lsim(sys, udata, simTime, xdata(1,:)');
wmaxTrain = max(xdata-xCompare);
wminTrain = min(xdata-xCompare);



%{
figure(2)
subplot(2,1,1)
plot(yCompare(:,1))
hold on
plot(ydata(:,1))
ylabel('Surface Tempearture/C')
subplot(2,1,2)
plot(yCompare(:,2))
hold on
plot(ydata(:,2))
ylabel('Intensity/ a.u.')
xlabel('Time Instance')
legend('Linear Model', 'Experimental Data')
%}

% opt = compareOptions('InitialCondition',ydata(1,:)');
opt = compareOptions('InitialCondition',xdata(1,:)');
figure(2)
compare(subIDdata, sys, opt)
xlabel('Time/s')
legend('Experimental Data', 'Linear Model')


%{
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VALIDATE MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load and clean the validation dataset
% dataFile = '/users/adbonzanini/Box Sync/Berkeley/Research/Experimental Data/Data_glass_metal/PI_Server_Out_2018-07-05_153054.004726-power_step_metal';
% dataValidation = csvread(dataFile,1, 0);
% dataValidation = dataValidation(1:end-50, :);

% Remove the rows of the dataset in which the oscilloscope malfunctioned
inds=find(dataValidation(:,8)>1e5);
dataValidation(inds,:)=[];
inds=find(dataValidation(:,7)>1e5);
dataValidation(inds,:)=[];

% Translate to the origin
y1ss = mean(dataValidation(1:10,3));     % Ts
y2ss = mean(dataValidation(1:10,27));     % Is
u1ss = mean(dataValidation(1:10,16));    % q
u2ss = mean(dataValidation(1:10,23));    % P

% Work with deviation variables
y1data = dataValidation(:,3)-y1ss;
y2data = dataValidation(:,27) - y2ss;
u1data = dataValidation(:,16)-u1ss;
u2data = dataValidation(:,23)-u2ss;


ydata = [y1data, y2data];
udata = [u1data, u2data];
subIDdataValidation = iddata(ydata, udata, 1.3);


yCompare = lsim(sys, udata, 0:1.3:1.3*size(udata,1)-1);
wmaxValidation = max(ydata-yCompare);
wminValidation = min(ydata-yCompare);
% yCompare = lsim(sys, udata, 0:1.3:1.3*size(udata,1)-1, ydata(1,:));
% figure(3)
% subplot(2,1,1)
% plot(yCompare(:,1))
% hold on
% plot(ydata(:,1))
% ylabel('Surface Tempearture/C')
% subplot(2,1,2)
% plot(yCompare(:,2))
% hold on
% plot(ydata(:,2))
% ylabel('Intensity/ a.u.')
% xlabel('Time Instance')
% legend('Linear Model', 'Experimental Data')

opt = compareOptions('InitialCondition',ydata(1,:)');
figure(3)
compare(subIDdataValidation, sys, opt)
xlabel('Time/s')
legend('Experimental Data', 'Linear Model')
%}

wmaxValidation = zeros(1,2);
wminValidation=zeros(1,2);
maxErrors = max([wmaxTrain;wmaxValidation], [], 1);
minErrors = min([wminTrain;wminValidation], [], 1);
disp(maxErrors)
disp(minErrors)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if overwriteData==1
    save('APPJmodelDAE', 'A', 'B', 'C', 'steadyStates', 'maxErrors', 'minErrors', 'xdata', 'udata', 'xCompare')
    warning('Datasets overwritten!')
    fprintf('\n')
end

cd(directory)
