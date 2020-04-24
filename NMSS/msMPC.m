% function [yTr,uOptSeq, ssPlot, Y, U] = msMPC(y0, wl, wu, Np, N, Q, R, PN)
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
% Date:         April 8 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize CasADi
rng(100,'twister')
import casadi.*

% Initial point
yi = y0;

% Robust horizon for multistage MPC
N_robust = 2;                       

%% Load plasma model
% sys = load('APPJmodelGlass10s2o');
sys = load('APPJ_NMSS').model;

% Dimensions
nx=size(sys.A,2);
nu=size(sys.B,2);
ny=size(sys.C,1);
nd=2;
Nrepeat = nx+ny+nu;

% Discrete uncertainty set
Wset = [wl, zeros(ny,1), wu];

% Matrices
A = sys.A;
B = sys.B;
C = sys.C;
Ew = sys.Ew;

%%
%%{
Fmat = []; % Np x Np
Phimat = []; % Np x (Np-1)
for j=1:Np
% Fmat
Fmat = [Fmat;A^j];
% Phimat
    PhiRow = [];
    for jj = j:-1:1
        PhiRow = [PhiRow, A^(j-1)*B];
    end
    if size(PhiRow,2)>(Np-1)*nu
        PhiRow = PhiRow(:,1:(Np-1)*nu);
    end

Phimat = [Phimat; PhiRow, zeros(size(A,1), (Np-1)*nu-size(PhiRow,2))];

end
%}
%%

%% Define Constraints
Y = [eye(2), [5;10]; - eye(2), [5; 6.36] ];
Y = Polyhedron('H', Y);
U = [eye(2), [10;10]; -eye(2), [1.5; 1.5]];
U = Polyhedron('H', U);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MULTI-STAGE NMPC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Set point(s) and time(s) at which the reference changes
ysp1 = [5;0];
tChange = 1000;
ysp2 = [0;0];
tChange2 = 9999;
ysp3 = [0;0];




yss = ysp1;
uss = [0;0];



%% Variables and model
% Declare the variables
x = MX.sym('x', nx);
u = MX.sym('u', nu);

ss = MX.sym('yssVar', ny+nu);
wNoise = MX.sym('wNoise', ny);


%% Define dynamics and cost as Casadi functions

% Dynamics and stage cost
xNext = A*x+B*u + Ew*ss(1:ny);%+vertcat(wNoise, zeros(nx-nd, 1));
y = C*x+wNoise;
Lstage = (y-ss(1:ny))'*Q*(y-ss(1:ny)) + (u-ss(ny+1:end))'*R*(u-ss(ny+1:end));

% Functions for nominal vs. real dynamics
F = Function('F', {x,u,wNoise,ss}, {xNext,Lstage},{'x','u', 'wNoise', 'ss'},{'xNext', 'Lstage'});


%% Offset-free and estimation matrices (if needed)

%Offset-free tracking parameters
Bd = zeros(nx,nx);
Cd = eye(ny,nx);
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


%% Solve the setpoint calculator multiparametrically to reduce computation time
% mpSP = spCalculator(Aaug, Bd, Cd, Y, U, Haug, sys);


%% Initialize MPC

% Scenarios
scenario_idx = [-1, 0, 1];     % Multiplier of the additive w(x,u) of the GP in each scenario

% Build the scenario matrix with all the combinations. Dimension (Ncases^Nrobust, Nrobust)
scenario_mat = combvec(scenario_idx, scenario_idx)';
N_scenarios = length(scenario_mat);

% Weights for cost function
w_i = (1/N_scenarios)*ones(N_scenarios,1);


% Initialize vectors to store the predicted trajectories for each scenario
yS = cell(ny, 1);
for j =1:ny
    yS{j} = zeros(N_scenarios, Np+1);
end

uS = cell(nu, 1);
for j =1:nu
    uS{j} = zeros(N_scenarios, Np);
end

Xvec = cell(N_scenarios,1);
Uvec = cell(N_scenarios,1);

% Initialize vectors to store the predicted inputs and outputs for one OCP loop
uopt = zeros(nu, Np);

% Initialize vectors to store the actual applied inputs and measured outputs
uOptSeq = zeros(nu, N);
fopt = zeros(N,1);
yTr = zeros(ny, N+1);


% Initialization
ssPlot = [ysp1(1);ysp1(2)];
ssPlot = [ssPlot, ssPlot];
% dhat = zeros(2,1);
xki = C\yi;
xhati = xki;
yTr(:, 1)= yi;


%% MPC Loop
Tstart = tic;

for k = 1:N
    xki = xhati;
    Jactual = 0;
    
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
       
        wReal = [1;0]+0*[normrnd(1., 0.3, [1,N+1]);normrnd(0., 0.3, [1,N+1])];
        
        
        X_NMSS = [];
        lbX_NMSS = [];
        ubX_NMSS = [];
        U_NMSS = [];
        lbU_NMSS = [];
        ubU_NMSS = [];

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
            
            if i<Np
                U_NMSS = [U_NMSS;Uk];
                lbU_NMSS = [lbU_NMSS;min(U.V)'];
                ubU_NMSS = [ubU_NMSS;max(U.V)'];
            end
           
            
            % Integrate until the end of the interval
            if i<=N_robust
                wPred = Wset(:,w_idx(i));
            else
                wPred = 0*Wset(:,1);
            end
            [Xk_end, Jstage] = F(Xk, Uk, wPred,[yss;uss]);

            % Yk_end = mtimes(C, Xk_end)+0*YGP
            J=J+w_i(n_sc)*Jstage;
            % Penalize abrupt changes
            %J = J + mtimes(mtimes((Uk-uopt[:,i]).T, RR), Uk-uopt[:,i]) #+ mtimes(mtimes((Yk_end-Yk).T, QQ), Yk_end-Yk) 
            
            g   = [g;Yk-C*Xk-wPred];
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
            
            X_NMSS = [X_NMSS;Xk];
            lbX_NMSS = [lbX_NMSS;-inf*ones(nx,1)];
            ubX_NMSS = [ubX_NMSS; inf*ones(nx,1)];
            
            
            
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
            
            
            %%%%%%%%%%%%%%%%%%%% Equality Constraints %%%%%%%%%%%%%%%%%%%%
            
            %%{
            g   = [g;Xk_end-Xk];
            lbg = [lbg; zeros(nx,1)];
            ubg = [ubg; zeros(nx,1)];
            %}
            
            
            
            
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
        % Terminal cost and constraints (Xk --> i+1)
        % Terminal Cost
        J = J + w_i(n_sc)*(Yk-yss)'*PN*(Yk-yss);
        
        % Equality constraint to make sure that Yk at the last step is equal to  C*Xk
        g = [g;Yk-C*Xk];
        lbg = [lbg;zeros(ny,1)];
        ubg = [ubg;zeros(ny,1)];
        
        % Equality constraints for NMSS  
        %{
        Xvec{n_sc} = X_NMSS;
        Uvec{n_sc} = U_NMSS;
            

        g=[g;X_NMSS - Fmat*xki-Phimat*U_NMSS];
        lbg=[lbg;zeros(size(X_NMSS,1),1)];
        ubg=[ubg;zeros(size(X_NMSS,1),1)];
        %}
        
        

     
        
        
    end
    %% NMSS Constraints
    %{
    for ii=N_scenarios
        for jj=2:Np+1
            idxStart = (jj-1)*Nrepeat+1;
            idxStart = idxStart+(ii-1)*((Np+1)*Nrepeat-1)-ii+1;

                fprintf('%s \n', [num2str(idxStart), '-->', strMat{idxStart}]);
            Xvec{ii} = [Xvec{ii}; w(idxStart:idxStart+nx-1)];

            if jj<=Np
            Uvec{ii} = [Uvec{ii}; w(idxStart+nx+ny:idxStart+nx+ny+nu-1)];
            end   
        end

        g=[g;Xvec{ii} - Fmat*xki-Phimat*Uvec{ii}];
        lbg=[lbg;zeros(size(Xvec{ii},1),1)];
        ubg=[ubg;zeros(size(Xvec{ii},1),1)];

    end
    
    fprintf('\n')
    for jj=1:n_sc
       %disp(Xvec{jj})
       disp(Uvec{jj})
    end
    fprintf('\n')
    %}
    %%



    
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

    Nbranch = length(scenario_idx); % branches per node
%     Nrepeat = length(w)/N_scenarios;
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
%     sol_opts = struct('ipopt.print_level',0, 'ipopt.max_cpu_time',10, "discrete", discrete);
    sol_opts = struct('discrete', discrete);
    sol_opts.ipopt.max_iter = 500;

    solver = nlpsol('solver', 'ipopt', prob, sol_opts);
    
    % Solve the NLP
    sol = solver('x0', w0, 'lbx',lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);

    w_opt = full(sol.x);
    J_opt = full(sol.f);
    
    
    % These only include the first scenario --> first case in scenario mat
    y1_opt = w_opt(nx+1:Nrepeat:Nrepeat*Np)';
    y2_opt = w_opt(nx+2:Nrepeat:Nrepeat*Np)';
    u1_opt = w_opt(nx+ny+1:Nrepeat:Nrepeat*(Np-1))';
    u2_opt = w_opt(nx+ny+2:Nrepeat:Nrepeat*(Np-1))';
    
    feasMeasure{k} = solver.stats.return_status;
    
    
    % Perform plant simulations for the next step using the plant model
    uopt = [u1_opt;u2_opt];
    
    uOptSeq(:,k) = uopt(:,1);
    fopt(k) = J_opt;
    
    %     Update intial condition. Can change the if statement to define 
    %     when the plant-model mismatch is introduced (e.g. glass-to-metal
    %     transition).
    
%     dist = [wFn(xki(1)); 0];
%     dhat = [Fpred(xki(1));0];
    
%     dhat = dist;
    modelOrder = size(sys.AX{1},2)-1;
    yPrev = xki(1:ny, 1);
    uPrev = uopt(:,1);
    xkiPrev = xki;
%     xkiPrev(ny*modelOrder+1:ny*modelOrder+2,1) = uPrev;
    
    xki = A*xki+B*uopt(:,1)+Ew*yss; %+[wReal(:,k); zeros(nx-nd,1)];
    yki = C*xki+wReal;
    
    yTr(1,k+1) =yki(1);
    yTr(2,k+1) =yki(2);
    
    
    % State Feedback
%     xki(ny+1:end) = xki(1:end-2);

    % xki(k+1) = [y(k+1), ..., u(k), ...,]
    %{
    xki(ny+1:end) = xkiPrev(1:end-2);
    xki(ny*modelOrder+1:ny*modelOrder+2,1) = uPrev;
    %}

    
    
    %{
    xki(ny+1:ny+ny,1) = yPrev;
    xki(ny*modelOrder+1:ny*modelOrder+2,1) = uPrev;
    %}
    
    xhati = xki;
%     L = [eye(ny);zeros(nx-ny, ny)];
%     xhati = A*xki+B*uopt(:,1) + L*(yki-C*xhati);

    %
    if k>=0 && k<tChange-1
        yspCase = ysp1;
    elseif k>=tChange-1 && k<tChange2-1
        yspCase = ysp2;
    elseif k>=tChange2-1
        yspCase = ysp3;
    end
        
    %{
    % Setpoint calculator 
    dhat = zeros(nx,1);
    sp_opt = spCalculator(yspCase, dhat, Aaug, Bd, Cd, Y, U, Haug, sys);
    
    % mpSP = spCalculator(Aaug, Bd, Cd, Y, U, Haug, sys);
%     sp_opt = mpSP([yspCase;dhat]);
    yss = sp_opt(1:2);
    uss = sp_opt(3:end);
    %}
 
    yss = yspCase;
    uss = B\((eye(size(A))-A)*(C\ysp1));
    ssPlot=[ssPlot, yspCase];    
    
end


% end

