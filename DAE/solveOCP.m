function [u_mpc, feas, v_opt, args] = solveOCP(solver, args, x, ytarget)

% Extract arguments
w0 = args.w0;
lbw = args.lbw;
ubw = args.ubw;
lbg = args.lbg;
ubg = args.ubg;
nx = args.nx;
ny = args.ny;
nu = args.nu;
nCEM = args.nCEM;
offset_x0 = args.offset_x0;
offset_CEM = args.offset_CEM;

% Update initial state
w0(offset_x0+1:offset_x0+nx) = x;
lbw(offset_x0+1:offset_x0+nx) = x;
ubw(offset_x0+1:offset_x0+nx) = x;

w0(offset_CEM+1:offset_CEM+nCEM) = ytarget;
lbw(offset_CEM+1:offset_CEM+nCEM) = ytarget;
ubw(offset_CEM+1:offset_CEM+nCEM) = ytarget;

% Solve the nmpc problem
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);
% Extract optimal solution
w_opt = full(sol.x);
u_mpc = w_opt(nx+nCEM+ny+1:nx+nCEM+ny+nu);

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

end