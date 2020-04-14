clear all

% Plot settings
Fontsize = 15;
Lwidth = 2;

% Get current directory
directory = pwd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD COLLECTED DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choose dataset
glass = 1;
metal = 0;
Dy1ss = 0;

% Choose if you want to overwrite the datasets
overwriteData =1;

% Sampling time
Tsampling = 1.3;

%Ignore the headers (i.e. start from row 1, column 0)

if glass ==1 && metal==0
     dataFile = 'PI_Server_Out_2019-10-16_141348.846364-modelID';
elseif metal ==1 && glass==0
    dataFile = [directory, '/'];
else
    error('Select a single dataset.')
end

data = csvread(dataFile,1, 0);

% Column legend (OLD)
%(1) time,(2) Tset, (3) Ts, (4) Ts2, (5) Ts3, (6) P, (7) Imax, (8) O777,
%(9) O845, (10) N391, (11) He706, (12) sum_int, (13) V, (14) F, (15) Q, 
%(16) Dsep, (17) x_pos, (18) y_pos, (19) T_emb, (20) V, (21) P_emb, (22) Prms,
% (23) D_c, (24) Rdel, (25) Is, (26) q_o, (27, 28) sig, (29) tm_el

% Column legend (NEW)
% (1) time,(2) Tset,(3) Ts,(4) Ts2,(5) Ts3, (6) P, (7) Imax, (8) Ip2p, 
% (9) O777, (10) O845, (11) N391, (12) He706, (13) sum_int, 
% (14, 15, 16, 17) *U_m --> (V, freq, q, dsep), (18) q_o, (19) D_c, (20) x_pos, 
% (21) y_pos, (22) T_emb, (23) Pset, (24) P_emb, (25) Prms, 
% (26) Rdel, (27) Is, (28, 29) sig --> (1 and 2), (30) subs_type, (31) Trot, 
% (32) tm_el

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLEAN DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Remove the last few rows (time points) when the experiment stopped but we kept collecting data
data = data(1:end-20,:);

% Normalize any intensity to make it of the same order of magnitude as T
data(:,27) = data(:,27)/10;

% Split dataset into training and validation. Exclude nonlinear region (at the end)
dataTrain = data(100:end, :);               %glass 100:1700; metal 100:1500
% dataValidation = data(1500-30:end-1100, :);  %glass 1700-30:end-800; metal 1500-30:end-1100
data = dataTrain;

% Remove the rows of the dataset in which the oscilloscope malfunctioned
inds=find(data(:,8)>1e5);
data(inds,:)=[];
inds=find(data(:,7)>1e5);
data(inds,:)=[];

% Plot the data to visualize it
figure(1)
subplot(4, 1, 1)
plot(data(:,3))
ylabel('T_s/C')
subplot(4, 1, 2)
plot(data(:,27))
ylabel('I_s/a.u.')
subplot(4, 1, 3)
plot(data(:,16))
ylabel('Flow/slm')
subplot(4, 1, 4)
plot(data(:,23))
hold on
plot(data(:,24))
ylabel('Power/W')
xlabel('Time Instance')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IDENTIFY THE MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Translate to the origin
y1ss = mean(data(1:10,3))+Dy1ss;    % Ts
y2ss = mean(data(1:10,27));     % Is
u1ss = mean(data(1:10,16));    % q
u2ss = mean(data(1:10,24));    % P
steadyStates = [y1ss, y2ss, u1ss, u2ss];
disp(['Steady States = ', num2str(steadyStates)]);


% Work with deviation variables
y1data = data(:,3) -y1ss;
y2data = data(:,27)-y2ss;
u1data = data(:,16)-u1ss;
u2data = data(:,23)-u2ss;

ydata = [y1data, y2data];
udata = [u1data, u2data];

% Create dataset
subIDdata = iddata(ydata, udata, Tsampling);
Ndata = subIDdata.N;


% Subspace ID algorithm(s)

% opt = ssestOptions('OutputWeight', [1,0;0,1]);
% sys = ssest(subIDdata,modelOrder, 'DisturbanceModel', 'none', 'Form', 'canonical', 'Ts',Tsampling);
% sys = n4sid(subIDdata, modelOrder, 'DisturbanceModel', 'none', 'Form', 'canonical', 'Ts',Tsampling);
% sys = ssregest(subIDdata,modelOrder, 'DisturbanceModel', 'none','Form', 'canonical', 'Ts',1.3);

%% Compare MSE and Fit Percent for different orders
disturbanceModel = 1;
orderVec = 2:9;
% Initialize vectors to store metrics
MSE = zeros(size(orderVec'));
fitPercent = zeros(size(orderVec,1), 2);
MSEarmax = zeros(size(orderVec'));
fitPercentarmax = zeros(size(orderVec,1), 2);
% Intialize counter
count=1;
for modelOrder = orderVec
    % State-space
    if disturbanceModel==1
        sys = n4sid(subIDdata, modelOrder, 'Form', 'canonical', 'Ts',Tsampling);
    else
        sys = n4sid(subIDdata, modelOrder, 'DisturbanceModel', 'none', 'Form', 'canonical', 'Ts',Tsampling);
    end
    MSE(count) = sys.Report.Fit.MSE;
    fitPercent(count,:) = sys.Report.Fit.FitPercent';
    
    % ARMAX
    NA=modelOrder*[1 0;0 1];
    NB=modelOrder*[1 1;1 1];
    NC=modelOrder*[0;0];   
    NK=[1 1;1 1];
    mARMAX = armax(subIDdata, [NA, NB, NC, NK]);
    MSEarmax(count) = sys.Report.Fit.MSE;
    fitPercentarmax(count,:) = sys.Report.Fit.FitPercent';
    
    % Update counter
    count = count+1;
end
disp([MSE, fitPercent])
disp([MSEarmax, fitPercentarmax])
%%

figure(2)
subplot(2,1,1)
hold on
plot(orderVec, MSE, 'b-o', 'MarkerSize', 5, 'Linewidth', Lwidth)
plot(orderVec, MSEarmax, 'r-o', 'MarkerSize', 5, 'Linewidth', Lwidth)
% legend('State Space', 'ARMAX')
ylabel('MSE')
set(gca, 'FontSize', Fontsize)
box on
subplot(2,1,2)
hold on
plot(orderVec, fitPercent(:,1), 'b-o', 'MarkerSize', 5, 'Linewidth', Lwidth)
plot(orderVec, fitPercentarmax(:,1), 'r-o', 'MarkerSize', 5, 'Linewidth', Lwidth)
% legend('State Space', 'ARMAX')
ylabel('Fit Percent for y_1')
xlabel('Model Order')
set(gca, 'FontSize', Fontsize)
box on

%% Balance small MSE and small order
modelOrder = 3;
sys = n4sid(subIDdata, modelOrder, 'Form', 'canonical', 'DisturbanceModel', 'none', 'Ts',Tsampling);
A = sys.A;
B = sys.B; 
C = sys.C; 
Ke = sys.K;

%% ARMAX Model
NA=modelOrder*[1 0;0 1];
NB=modelOrder*[1 1;1 1];
NC=modelOrder*[0;0];   
NK=[1 1;1 1];
mARMAX = armax(subIDdata, [NA, NB, NC, NK]);
AX = mARMAX.A;
BX = mARMAX.B;
CX = mARMAX.C;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERIFY THE MODEL GRAPHICALLY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
simTime = 0:Tsampling:Tsampling*(size(udata,1)-1);
yCompare = lsim(sys, udata, simTime);
wmaxTrain = max(ydata-yCompare);
wminTrain = min(ydata-yCompare);

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








%%
% Compare
opt = compareOptions('InitialCondition',zeros(modelOrder,1));
figure(3)
compare(subIDdata, sys, mARMAX)
xlabel('Time/s')
legend('Experimental Data', 'SS Model', 'ARMAX Model')
%%


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
cd ..
if overwriteData==1
    if glass ==1
        save('APPJssGlass', 'A', 'B', 'C', 'Ke', 'steadyStates', 'maxErrors', 'minErrors')
        save('APPJarmaxGlass', 'AX', 'BX', 'CX', 'steadyStates', 'maxErrors', 'minErrors')
    elseif metal==1
        save('APPJmodelMetal10s2o', 'A', 'B', 'C', 'steadyStates', 'maxErrors', 'minErrors')
    end
    warning('Datasets overwritten!')
    fprintf('\n')
end

cd(directory)
