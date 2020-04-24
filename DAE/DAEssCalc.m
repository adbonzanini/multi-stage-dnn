function [xdss, xass, dss, uss] = DAEssCalc(Tref,dref, reduced)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates the steady states for the DAE model corresponding to the
% reference temperature Tref in oC and reference disturbance
% (tip-to-surface separation distance) dref in mm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import casadi.*
% Run initialization to define variables, parameters, etc
initializePlant;

% Initial reference values (used to calculate the initial steady-state)
dref = dref;
if reduced == 0
    yref = [(Tref+273.0)/300.0];
else
    yref = Tref;
end


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



end

