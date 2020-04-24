function [U_mpc, Feas, V_opt] = solveSamplesMPC(solver, args, X0)

% Get number of samples
N = size(X0,1);

% Solve nmpc problem for each sample
U_mpc = zeros(N,args.nu);
Feas = zeros(N,1);
V_opt = zeros(N,1);

for i = 1:N
    % Print start statement
    fprintf('Sample %g of %g...', i, N)
    tic
    
    % Call function to solve problem and return data
    [U_mpc(i,:), Feas(i), V_opt(i), args] = solveOCP(solver, args, X0(i,1:2), X0(i,3));
    
    % Print end statement
    fprintf('took %g seconds\n', toc)
end

end