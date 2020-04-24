function [solver, args, Y, U] = msMPCsolver(yi, sys, currentCEM, CEMtarget, wl, wu, Np, N, Q, R, PN, GPtraining)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary: Function that implements a multi-stage MPC strategy
% y0: initial states
% wl, wu: uncertainty bounds w \in [wl, wu]
% Np: prediction horizon
% N: simulation horizon
% Q, R, and PN: Cost matrices


% Additional information
% Need a plasma model to run. Matrices A, B and C are loaded within the
% function
% Setpoint trajectory fixed within the function
% Nr: Robust horizon = 2 defined within the function
% Discretization of uncertainty such that w = {wl, 0, wu}

% Written by:   Angelo D. Bonzanini
% Date:         April 20 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize CasADi
rng(100,'twister')
import casadi.*

% Initial point
% yi

% Robust horizon for multistage MPC
N_robust = 2;                       

%% Load plasma model
% sys passed as an input

% Dimensions
nx=size(sys.A,2);
nu=size(sys.B,2);
ny=size(sys.C,1);
nd=2;
nCEM = size(CEMtarget,1);

% Discrete uncertainty set
Wset = [wl, zeros(ny,1), wu];

% Matrices
A = sys.A;
B = sys.B;
C = sys.C;

% CEM Parameters
Kcem = 0.75; % if T>=35 oC


%% Define Constraints
Y = [eye(2), [5;8.5]; -eye(2), [10; sys.steadyStates(2)]];
Y = Polyhedron('H', Y);
U = [eye(2), [5;5]; -eye(2), [1.5; 1.5]];
U = Polyhedron('H', U);

%% Unpack the GP training object
myKrigingMat = GPtraining.myKrigingMat;
Xtest = GPtraining.Xtest;
lag = GPtraining.lag;
[GP, ~] = uq_evalModel(myKrigingMat, Xtest);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MULTI-STAGE NMPC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set point(s) and time(s) at which the reference changes
dref = 4.0; % Separation distance reference
ysp1 = [0;0];
tChange = 1000;
ysp2 = [0;0];
tChange2 = 9999;
ysp3 = [0;0];


yss = ysp1;
[xd0, xa0, d0, uss] = DAEssCalc(yi(1)+sys.steadyStates(1),dref, 0);



%% Variables and model
% Declare the variables
x = MX.sym('x', nx);
u = MX.sym('u', nu);

ss = MX.sym('yssVar', ny+nu);
wNoise = MX.sym('wNoise', ny);


%% Define dynamics and cost as Casadi functions

% Dynamics and stage cost
xNext = A*x+B*u+wNoise;
y = C*x;

% Tracking stage cost
Lstage = (y-ss(1:ny))'*Q*(y-ss(1:ny)) + (u-ss(ny+1:end))'*R*(u-ss(ny+1:end));

% CEM stage cost
CEMstage =  Kcem^(43-(x(1)+wNoise(1)+sys.steadyStates(1)));
Lfn = Function('Lfn', {x, wNoise}, {CEMstage}, {'x', 'wNoise'}, {'CEMstage'});

% Functions for nominal vs. real dynamics
F = Function('F', {x,u,wNoise,ss}, {xNext,Lstage},{'x','u', 'wNoise', 'ss'},{'xNext', 'Lstage'});


%% Offset-free and estimation matrices (if needed)

%Offset-free tracking parameters
Bd = zeros(nx,nd);
Cd = eye(ny,nd);
Haug = eye(ny,ny);
Aaug = [eye(size(A))-A, -B; Haug*C, zeros(ny, nu)];

%{
% Steady-state Kalman Gain
Adist = [A, Bd; zeros(2,2), eye(2,2)];
Bdist = [B;zeros(nx, nu)];
Cdist = [C, Cd];
Qnoise = 10*diag([0.1, 0.1, 0.1, 0.1]);
Rnoise = 10*diag([0.01, 0.01]);
[Pinf, ~, Kinf] = dare(Adist', Cdist', Qnoise, Rnoise);
Kinf = -Kinf;
% LQR gain
[Plqr, ~, Klqr] = dare(A, B, Q, R);
Klqr = -Klqr;
%}


%% Initialize MPC

% Scenarios
scenario_idx = [-1, 0, 1];     % Multiplier of the additive w(x,u) of the GP in each scenario

% Build the scenario matrix with all the combinations. Dimension (Ncases^Nrobust, Nrobust)
scenario_mat = combvec(scenario_idx, scenario_idx)';
N_scenarios = length(scenario_mat);

% Weights for cost function
w_i = (1/N_scenarios)*ones(N_scenarios,1);


% Initialization
% dhat = zeros(2,1);
xki = [xd0(2)*300-273;xa0(1)]-sys.steadyStates(1:2)';
xhati = xki;



%% MPC Loop
xki = xhati;

fprintf('\n\n################################# NEW OPTIMIZATION #################################\n\n')

%   At each step k the entire OCP for all scenarios is solved!
%   Therefore, we need to intialize empty variables for each
%   step k.

% Start with an empty NLP
w=[];    %Array of all the variables we will be optimizing over
w0 = [];
lbw = [];
ubw = [];
discrete = [];
J = 0;
g=[];
lbg = [];
ubg = [];
strMat = {};

% Check for errors
checkSc = [];
feasMeasure = cell(N,1);
%     disp(xhati) 


% MPC LOOP FOR DIFFERENT SCENARIOS - ROBUST HORIZON = 2
for n_sc =1:length(scenario_mat)
    sc_vec = scenario_mat(n_sc, :)';
    w_idx = 1-min(min(scenario_mat))+sc_vec;

    rng(n_sc)

    wReal = [3; 0]+0*[normrnd(1., 0.3, [1,N+1]);normrnd(0., 0.3, [1,N+1])];

    %     "Lift" initial conditions. Note that the initial node
    %     is the same for all scenarios, so the double index is not
    %     required.

    Xk = MX.sym(char(join(["X0","_",string(n_sc)])), nx);
    strMat = [strMat;{char(join(["X0","_",string(n_sc)]))}];
    for jj=1:nx-1
        strMat = [strMat;strMat{end}];
    end
    w = [w;Xk];
    lbw = [lbw;xhati];
    ubw = [ubw;xhati];
    w0 = [w0;zeros(nx,1)];
    discrete =[discrete;zeros(nx,1)];

    Yk = MX.sym(char(join(['Y0','_',string(n_sc)])), ny);
    strMat = [strMat;{char(join(['Y0','_',string(n_sc)]))}];
    for jj=1:ny-1
        strMat = [strMat;strMat{end}];
    end
    w = [w;Yk];
    lbw = [lbw; -inf*ones(ny,1)];
    ubw = [ubw; inf*ones(ny,1)];
    w0 = [w0; zeros(ny,1)];
    discrete =[discrete; zeros(ny,1)];


    CEMvar = MX.sym('CEMcurr', nCEM);
    strMat = [strMat;{char(join(['CEMvar0','_',string(n_sc)]))}];
    w = [w;CEMvar];
    lbw = [lbw; currentCEM];
    ubw = [ubw; currentCEM];
    w0 = [w0; currentCEM];
    discrete =[discrete; zeros(nCEM,1)];

    % Optimal Control Problem - Open Loop Optimization

    % Formulate the NLP
    for i = 1:Np

        % New NLP variable for the control
        Uk = MX.sym(char(join(['U_',string(i),'_',string(n_sc)])), nu);
        strMat = [strMat;{char(join(['U_',string(i),'_',string(n_sc)]))}];
        for jj=1:nu-1
            strMat = [strMat;strMat{end}];
        end
        w   = [w;Uk];
        lbw = [lbw; min(U.V)'];
        ubw = [ubw; max(U.V)'];
        w0  = [w0; zeros(nu,1)];
        discrete =[discrete;zeros(nu,1)];


        % Integrate until the end of the interval
        if i<=N_robust
            wPred = Wset(:,w_idx(i));
            [Xk_end, ~] = F(Xk, Uk, 0*GP'+wPred,[yss;uss]);
        else
            [Xk_end, ~] = F(Xk, Uk, [0;0],[yss;uss]);
        end

        Jstage = Lfn(Xk, wPred);

        % Yk_end = mtimes(C, Xk_end)+0*YGP
        J=J+w_i(n_sc)*Jstage;
        %J = J + Jstage;
        % Penalize abrupt changes
        %J = J + mtimes(mtimes((Uk-uopt[:,i]).T, RR), Uk-uopt[:,i]) #+ mtimes(mtimes((Yk_end-Yk).T, QQ), Yk_end-Yk) 

        g   = [g;Yk-C*Xk];
        lbg = [lbg;zeros(ny,1)];
        ubg = [ubg;zeros(ny,1)];


        % New NLP variable for state at end of interval
        Xk = MX.sym(char(join(['X_',string(i+1),'_',string(n_sc)])), nx);
        strMat = [strMat;{char(join(['X_',string(i+1),'_',string(n_sc)]))}];
        for jj=1:nx-1
            strMat = [strMat;strMat{end}];
        end
        w   = [w;Xk];
        lbw = [lbw;-inf*ones(nx,1)];
        ubw = [ubw; inf*ones(nx,1)];
        w0  = [w0;zeros(nx,1)];
        discrete =[discrete;zeros(nx,1)];


        Yk = MX.sym(char(join(['Y_',string(i+1),'_', string(n_sc)])), ny);
        strMat = [strMat;{char(join(['Y_',string(i+1),'_', string(n_sc)]))}];
        for jj=1:ny-1
            strMat = [strMat;strMat{end}];
        end
        w   = [w;Yk];
        ubw = [ubw; max(Y.V)'];
        lbw = [lbw; min(Y.V)'];
        w0  = [w0;zeros(ny,1)];
        discrete =[discrete;zeros(ny,1)];

        g   = [g;Xk_end-Xk];
        lbg = [lbg; zeros(nx,1)];
        ubg = [ubg; zeros(nx,1)];

        %{
        if i==2
        Ybin = MX.sym(char(join(['Ybin_',string(i),'_', string(n_sc)])), npoly,1);
        strMat = [strMat;{char(join(['Y_',string(i),'_', string(n_sc)]))}]; 
        w   = [w;Ybin];
        lbw = [lbw;zeros(npoly,1)];
        ubw = [ubw; ones(npoly,1)];
        w0  = [w0;zeros(npoly,1)];
        discrete =[discrete;ones(npoly,1)];
        sumCon = 0;
        M=50;
            for ii=1:npoly
                strMat = [strMat;strMat{end}]; 
                g   = [g;Cinf_next(ii).A*Xk-Cinf_next(ii).b-M*(1-Ybin(ii))];
                lbg = [lbg; -inf*ones(length(Cinf_next(ii).b),1)];
                ubg = [ubg; zeros(length(Cinf_next(ii).b),1)];
                sumCon = sumCon + Ybin(ii);
            end
            g = [g; sumCon-1];
            lbg = [lbg;0];
            ubg = [ubg;0];

        end
        %}


    end

    % Equality constraint to make sure that Yk at the last step is equal to  C*Xk
    g = [g;Yk-C*Xk];
    lbg = [lbg;zeros(ny,1)];
    ubg = [ubg;zeros(ny,1)];

    % Terminal cost and constraints (Xk --> i+1)
    % Terminal Cost
%    J = J + w_i(n_sc)*(Yk-yss)'*PN*(Yk-yss);
%         J_end = Kcem^(43-(Xk(1)+sys.steadyStates(1)));
    J_end = Lfn(Xk, wPred);
    J = J+w_i(n_sc)*J_end;

end

Jcon = J + CEMvar;
J = (Jcon-CEMtarget).^2;

% Add terminal inequality constraint
%{
g=[g;Jcon];
lbg = [lbg; 0];
ubg = [ubg; CEMtarget];
%}




conCheck0=[];
conCheck1=[];    

% First split - Find all the U_0's
%%
u0idx = [];
u1idx = [];
for l=1:length(strMat)
    st0= strfind(strMat{l}, 'U_ 1');
    st1= strfind(strMat{l}, 'U_ 2');
    if isempty(st0)==0
        u0idx =[u0idx;l];
    end
    if isempty(st1)==0
        u1idx =[u1idx;l];
    end
end
%%

for con_idx = 1:nu:length(u0idx(1:end-nu))
    for iu=0:nu-1
        g = [g;w(u0idx(con_idx)+iu)-w(u0idx(con_idx+nu)+iu)];
        conCheck0 = [conCheck0;w(u0idx(con_idx)+iu)-w(u0idx(con_idx+nu)+iu)];
    end

    lbg = [lbg;zeros(nu,1)];
    ubg = [ubg;zeros(nu,1)];
end

%%

% Second split

% Group second scenarios based on the first split and then examine each
% group of branches individually
splitIdx = cell(1);
for l = 1:length(scenario_idx)
    splitIdx{l,1} = find(scenario_mat(:,1)==scenario_idx(l));
end

% The scenarios U_2_splitIdx{l} should be equal, since they stem from the
% same parent node

for l = 1:length(scenario_idx)
    for ll =1:length(splitIdx{l})-1
        for iu=0:nu-1
            g = [g;w(u1idx(splitIdx{l}(ll)))-w(u1idx(splitIdx{l}(ll)+nu+iu))];
            conCheck1 = [conCheck1;w(u1idx(splitIdx{l}(ll)))-w(u1idx(splitIdx{l}(ll)+nu+iu))];
        end
        lbg = [lbg;zeros(nu,1)];
        ubg = [ubg;zeros(nu,1)];
    end
end



%%
% Create an NLP solver
prob = struct('f', J, 'x', w, 'g',g);
sol_opts = struct();
%     sol_opts = struct('discrete', discrete);
sol_opts.discrete = discrete;
sol_opts.ipopt.max_iter = 500;
%     sol_opts.ipopt.print_level = 0;


solver = nlpsol('solver', 'ipopt', prob, sol_opts);

% Store arguments
args.w0 = w0;
args.lbw = lbw;
args.ubw = ubw;
args.lbg = lbg;
args.ubg = ubg;
args.offset_mpc = nx+ny+nu;
args.offset_x0 = 0;
args.offset_CEM = nx;
args.nx = nx;
args.ny = ny;
args.nu = nu;
args.nCEM = nCEM;
args.warm_start = 1;



end

