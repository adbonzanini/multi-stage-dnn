function [u_mpc, feas, v_opt, args, u_mpc1] = solveOCP(solver, args, x, params)

% Extract arguments
w0 = args.w0;
lbw = args.lbw;
ubw = args.ubw;
lbg = args.lbg;
ubg = args.ubg;
nx = args.nx;
ny = args.ny;
nu = args.nu;
nw = args.nw;
nCEM = args.nCEM;
offset_x0 = args.offset_x0;
offset_CEM = args.offset_CEM;
offset_w = args.offset_w;
offset_mpc = args.offset_mpc;
offset_end = args.offset_end;
%offset_scenario = args.offset_scenario;
Np = args.Np;
N_robust = args.N_robust;



% Scenarios
scenario_idx = [-1, 0, 1];     % Multiplier of the additive w(x,u) of the GP in each scenario

% Build the scenario matrix with all the combinations. Dimension (Ncases^Nrobust, Nrobust)
scenario_mat = combvec(scenario_idx, scenario_idx)';

% Update uncertainty bounds
Wset = [-params(2:3)', zeros(2,1), params(2:3)'];

for n_sc =1:length(scenario_mat)
    sc_vec = scenario_mat(n_sc, :)';
    w_idx = 1-min(min(scenario_mat))+sc_vec;
    
    
    Ncurr = (n_sc-1)*(Np*offset_mpc+nCEM+offset_end);
    
    % Update initial state for all scenarios
    Ncurr_x0 = Ncurr+offset_x0;
    w0(Ncurr_x0+1:Ncurr_x0+nx) = x;
    lbw(Ncurr_x0+1:Ncurr_x0+nx) = x;
    ubw(Ncurr_x0+1:Ncurr_x0+nx) = x;


    % Update initial CEM for all scenarios
    Ncurr_cem = Ncurr+offset_CEM;
    w0(Ncurr_cem+1:Ncurr_cem+nCEM) = params(1);
    lbw(Ncurr_cem+1:Ncurr_cem+nCEM) = params(1);
    ubw(Ncurr_cem+1:Ncurr_cem+nCEM) = params(1);

    
        % Update branches for all scenarios and all steps in robust horizon
        for i=1:Np
        Ncurr_w = Ncurr+offset_w+(i-1)*Np+min(1, i-1)*offset_end;
        if i<=N_robust
            w0(Ncurr_w+1:Ncurr_w+nw) = [Wset(1,w_idx(i));Wset(2,w_idx(i))];
            lbw(Ncurr_w+1:Ncurr_w+nw) = [Wset(1,w_idx(i));Wset(2,w_idx(i))];
            ubw(Ncurr_w+1:Ncurr_w+nw) = [Wset(1,w_idx(i));Wset(2,w_idx(i))];
%         else
%            w0(Ncurr_w+1:Ncurr_w+nw) = zeros(nw,1);
%            lbw(Ncurr_w+1:Ncurr_w+nw) = zeros(nw,1);
%            ubw(Ncurr_w+1:Ncurr_w+nw) = zeros(nw,1); 
        end
    end
 
end

% Solve the nmpc problem
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);
% Extract optimal solution
w_opt = full(sol.x);


u_mpc = w_opt(nx+nCEM+ny+1:nx+nCEM+ny+nu);
u_mpc1 = w_opt(offset_mpc+nx+nCEM+ny+1:offset_mpc+nx+nCEM+ny+nu);

% Check feasibility
ipopt_stats = solver.stats();
if strcmp(ipopt_stats.return_status,'Solve_Succeeded')
    feas = 1;
else
    feas = -1;
end

% Get objective value
v_opt = full(sol.f);

% Warm start, if specified
if args.warm_start == 1
    args.w0 = w_opt;
end
args.w0 = w0;
args.lbw = lbw;
args.ubw=ubw;
args.lbg=lbg;
args.ubg=ubg;
% fprintf('\n')
% disp([w0(1:112,:), [1:length(w0(1:112,:))]'])
end