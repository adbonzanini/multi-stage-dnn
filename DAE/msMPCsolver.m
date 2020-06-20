function [solver, args, Y, U] = msMPCsolver(yi, sys, currentCEM, CEMtarget, wl, wu, Np, Q, R, PN, GPtraining, Kcem, GPinPredictionIdx)
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
N = 1; % (not needed)

%% Load plasma model
% sys passed as an input

% Dimensions
nx=size(sys.A,2);
nu=size(sys.B,2);
ny=size(sys.C,1);
nw=2;
nCEM = size(CEMtarget,1);

% Discrete uncertainty set
Wset = [wl, zeros(ny,1), wu];

% Matrices
A = sys.A;
B = sys.B;
C = sys.C;

% CEM Parameters given in inputs
% Kcem = 0.75; % if T>=35 oC



%% Define Constraints
y1Con = [42.5;30]-sys.steadyStates(1);
y2Con = [10;0]-sys.steadyStates(2);
Y = [eye(2), [y1Con(1);y2Con(1)]; -eye(2), [-y1Con(2); -y2Con(2)]];
Y = Polyhedron('H', Y);
U = [eye(2), [5;5]; -eye(2), [1.5; 1.5]];
U = Polyhedron('H', U);

%% Unpack the GP training object
gprMdl1 = GPtraining.gprMdl1;
gprMdl2 = GPtraining.gprMdl2;
kfcn = GPtraining.kfcn; % "custom" SE kernel (for prediction within the OCP)
Xtrain = GPtraining.Xtrain;
Ytrain = GPtraining.Ytrain;
Xtest = GPtraining.Xtest;
lag = GPtraining.lag;


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
[xd0, xa0, d0, uss] = DAEssCalc(yi(1)+sys.steadyStates(1),dref,0);



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
Bd = zeros(nx,nw);
Cd = eye(ny,nw);
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

%     wReal = [3; 0]+0*[normrnd(1., 0.3, [1,N+1]);normrnd(0., 0.3, [1,N+1])];

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

    % Create CEM variable (parameter 1)
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
        w0  = [w0; 0.09*ones(nu,1)];
        discrete =[discrete;zeros(nu,1)];

        
        % Create wPred variable (parameters 2 and 3)
        wPred = MX.sym(char(join(['wPred_',string(i),'_',string(n_sc)])), nw);
        strMat = [strMat;{char(join(['wPred', '_', string(i),'_',string(n_sc)]))}];
        for jj=1:nw-1
            strMat = [strMat;strMat{end}];
        end
        w = [w;wPred];
        % Change bounds depending on robust horizon
        if i<=N_robust-GPinPredictionIdx
            lbw = [lbw; Wset(1,w_idx(i));Wset(2,w_idx(i))];
            ubw = [ubw; Wset(1,w_idx(i));Wset(2,w_idx(i))];
            w0 = [w0; Wset(1,w_idx(i));Wset(2,w_idx(i))];
            discrete =[discrete; zeros(nw,1)]; 
            
        elseif i>1 && i<=N_robust
            % Update test dataset
            XtestPred = [(Xk-xki)', Uk'];
            
            theta1 = gprMdl1.KernelInformation.KernelParameters;  %(sigmaL, sigmaF)
            KK = kfcn(Xtrain, Xtrain, theta1);
            KKs = kfcn(Xtrain, Xtest, theta1);
            KKss = kfcn(Xtest, Xtest, theta1);
            covGP1 = KKss - (KKs'/KK)*KKs;
            wGP1 = (KKs'/KK)*Ytrain(:,1)+3*sqrt(covGP1);
            
            %wGP2 = predict(gprMdl2, XtestPred); %Not needed since we only care about scenarios in x1
            Wset = [-[wGP1;0], zeros(ny,1), [wGP1;0]];
            
            % Update bounds
            lbw = [lbw; Wset(1,w_idx(i));Wset(2,w_idx(i))];
            ubw = [ubw; Wset(1,w_idx(i));Wset(2,w_idx(i))];
            w0 = [w0; Wset(1,w_idx(i));Wset(2,w_idx(i))];
            discrete =[discrete; zeros(nw,1)]; 
        else
            lbw = [lbw; zeros(nw, 1)];
            ubw = [ubw; zeros(nw, 1)];
            w0 = [w0; zeros(nw, 1)];
            discrete =[discrete; zeros(nw,1)];
        end
            [Xk_end, ~] = F(Xk, Uk, wPred,[yss;uss]);
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
    J_end = Lfn(Xk, [0;0]);
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



offset_mpc = nx+ny+nu+nw;
offset_end = nx+ny;

conCheck0=[];
conCheck1=[]; 


% Non-anticipativity constraint for k=0 (k=1 in MATLAB)
u1b1_all = []; %u(1) of 1st branching stage - all of them stacked
u2b1_all = []; %u(2) of 1st branching stage - all of them stacked
u1b2_all = []; %u(1) of 2nd branching stage - all of them stacked
u2b2_all = []; %u(2) of 2nd branching stage - all of them stacked

for jj = 1:N_scenarios
    Ncurr = (jj-1)*(Np*offset_mpc+nCEM+offset_end);  % for Np=4 --> Ncurr = 0, 37, 74, ...
    offset_u1 = Ncurr+nx+ny+nCEM;
    offset_u2 = offset_u1 + nu+nw+nx+ny;
    
    % Group all u at k=1 (k=0 in python)
    u1b1_all = [u1b1_all;w(offset_u1+1)];
    u2b1_all = [u2b1_all;w(offset_u1+nu)];
    
    if jj>1
        g = [g;
             u1b1_all(jj-1)-u1b1_all(jj);
             u2b1_all(jj-1)-u2b1_all(jj)];
         
        lbg = [lbg;zeros(nu,1)];
        ubg = [ubg;zeros(nu,1)];
        
        conCheck0 = [conCheck0;
                     u1b1_all(jj-1)-u1b1_all(jj);
                     u2b1_all(jj-1)-u2b1_all(jj)];
    end
    
    
    % Group all u at k=2 (k=1 in python)
    u1b2_all = [u1b2_all;w(offset_u2+1)];
    u2b2_all = [u2b2_all;w(offset_u2+nu)];
   
end

%{
% Find branching stages (3 sets of constraints because there are 3 nodes at k=2 (or k=1 in Python))
idx_b2{1} = find(scenario_mat(:,1)==-1);
idx_b2{2} = find(scenario_mat(:,1)==0);
idx_b2{3} = find(scenario_mat(:,1)==1);


for ii = 1:length(idx_b2)
    for kk=2:length(idx_b2{ii})
        g = [g;
            u1b2_all(idx_b2{ii}(kk-1))-u1b2_all(idx_b2{ii}(kk));
            u2b2_all(idx_b2{ii}(kk-1))-u2b2_all(idx_b2{ii}(kk))];

     conCheck1 = [conCheck1;
                 u1b2_all(idx_b2{ii}(kk-1))-u1b2_all(idx_b2{ii}(kk));
                  u2b2_all(idx_b2{ii}(kk-1))-u2b2_all(idx_b2{ii}(kk))];
    lbg = [lbg;zeros(nu,1)];
    ubg = [ubg;zeros(nu,1)];
    end 
end
%}



%{
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
%}


%%
% Create an NLP solver
prob = struct('f', J, 'x', w, 'g',g);
sol_opts = struct();
sol_opts.discrete = discrete;
sol_opts.ipopt.max_iter = 500;
sol_opts.ipopt.print_level = 1;
sol_opts.verbose = 0;


solver = nlpsol('solver', 'ipopt', prob, sol_opts);

% Store arguments
args.w0 = w0;
args.lbw = lbw;
args.ubw = ubw;
args.lbg = lbg;
args.ubg = ubg;
args.wvar = w;

args.nx = nx;
args.ny = ny;
args.nu = nu;
args.nw = nw;
args.nCEM = nCEM;

args.offset_mpc = nx+ny+nu+nw;
args.offset_scenario = args.offset_mpc*Np;
args.offset_x0 = 0;
args.offset_CEM = nx+ny;
args.offset_w = nx+ny+nCEM+nu; %Need to add nCEM only once!
args.offset_end = nx+ny;
args.Np = Np;
args.N_robust = N_robust;

args.warm_start = 1;



end

