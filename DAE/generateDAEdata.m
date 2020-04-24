%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates input-output data from high-fidelity DAE model so that it can
% be use for subspace identification to derive a state-space model. Then,
% the state-space model can be used as the MPC model and the DAE model as
% the plant model.
%
% Written by: Angelo D. Bonzanini
% Last edited: April 15 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
import casadi.*
initializePlant;

%% User inputs
% Overwrite data
overwrite = 0;

% Initial reference values (used to calculate the initial steady-state)
dref = [4.0];
yref = [(37.0+273.0)/300.0];
DT = 15;


% Input vector [P_set; q];
P_set_vals = [0.5:0.6:4.7];
q_vals = [1.5:0.5:5];



uSequence = combvec(P_set_vals, q_vals);
nComb = size(uSequence,2);
rng(0)
shuffleIdx = randi([1, nComb], 2, nComb);
uSequence = uSequence(shuffleIdx);
uSequence = imresize(uSequence, [2, nComb*DT], 'nearest');
Nsim = size(uSequence,2);

figure(1)
subplot(2,1,1)
stairs(uSequence(1,:))
subplot(2,1,2)
stairs(uSequence(2,:))


%% Calculate the initial steady state

% Control bounds
u_min = [0, 1.5];
u_max = [4.7, 5];
u_init = [2.35, 3.25];

% Differential state bounds
%Path bounds
xD_min = [0.0, 1.0, -inf, 0.0];
xD_max = [inf, (47.0+273.0)/300.0, inf, 0.0];
%Initial guess for differential states
xD_init = [(40.0+273.0)/300.0, (40.0+273.0)/300.0, 1.0, 0.0]; % needs to be specified for every time interval

% Algebraic state bounds and initial guess
xA_min =  [0.0, 0.0, 0.0, 0.0, 3.0];
xA_max =  [inf, 4.7, inf, inf, 5.0];
xAi_min = [0.0, 0.0, 0.0, 0.0, 0.0];
xAi_max = [inf, inf, inf, inf, inf];
xAf_min = [0.0, 0.0, 0.0, 0.0, 0.0];
xAf_max = [inf, inf, inf, inf, inf];
xA_init = [4.0, 4.5, 16.0, (40.0+273.0)/300.0, 4.0];

% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];

% New NLP variables
Xss = MX.sym('Xss', ndstate);
w = [w, {Xss}];
lbw = [lbw, xD_min];
ubw = [ubw, xD_max];
w0 = [w0, xD_init];
Zss = MX.sym('Xss', nastate);
w = [w, {Zss}];
lbw = [lbw, xA_min];
ubw = [ubw, xA_max];
w0 = [w0, xA_init];
Uss = MX.sym('Uss', ninput);
w = [w, {Uss}];
lbw = [lbw, u_min];
ubw = [ubw, u_max];
w0 = [w0, u_init];
Yss = MX.sym('Yss', noutput);
w = [w, {Yss}];
lbw = [lbw, yref'];
ubw = [ubw, yref'];
w0 = [w0, yref'];
Pss = MX.sym('Pss', nparam);
w = [w, {Pss}];
lbw = [lbw, dref(1)];
ubw = [ubw, dref(1)];
w0 = [w0, dref(1)];

% Add steady state equations
g = [g, {f_x_fcn(0.,Xss,Zss,Uss,Pss)}];
g = [g, {f_z_fcn(0.,Xss,Zss,Uss,Pss)}];
g = [g, {Yss-houtfcn(0.,Xss,Zss,Uss,Pss)}];
lbg = [lbg, zeros(1,ndstate+nastate+noutput)];
ubg = [ubg, zeros(1,ndstate+nastate+noutput)];

% Objective
unom = u_init';
J = J + (Yss - yref)'*(Yss - yref);
J = J + (Uss - unom)'*(Uss - unom);

% Create an NLP solver
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob);

% Solve the NLP
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw,...
            'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

% Extract solution
offset = 1;
xdss = w_opt(offset:offset+ndstate-1);
offset = offset + ndstate;
xass = w_opt(offset:offset+nastate-1);
offset = offset + nastate;
uss = w_opt(offset:offset+ninput-1);
offset = offset + ninput;
yss = w_opt(offset:offset+noutput-1);
offset = offset + noutput;
dss = w_opt(offset:offset+nparam-1);

% Initialize
xdVec = zeros(size(xdss,1), Nsim);
xdVec(:,1) = xdss;
xaVec = zeros(size(xass,1), Nsim);
xaVec(:,1) = xass;

for j = 1:Nsim-1
    Fsim = plantSimulator(xdVec(:,j), uSequence(:,j), dss, xaVec(:,j));
    xdVec(:,j+1) = full(Fsim.xf);
    xaVec(:,j+1) = full(Fsim.zf);
end

%%
figure(2)
hold on
plot(xdVec(2,:)*300)
xlim([1, 100])


if overwrite==1
    save('Model_ID/DAE_data', 'xdVec', 'xaVec', 'uSequence')
end
