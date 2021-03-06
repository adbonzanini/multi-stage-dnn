function [U_mpc, Feas, V_opt, U_mpc1] = solveSamplesMPC(solver, args, X0)

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
    [U_mpc(i,:), Feas(i), V_opt(i), args, U_mpc1, updateFlag] = solveOCP(solver, args, X0(i,1:2), X0(i,3:end));
    
    % Print end statement
    fprintf('\n \n Fesibility %g \n \n', Feas(i))
    fprintf('took %g seconds\n', toc)
    if updateFlag==1
        warning('Bounds updated through external GP')
    end
end

end