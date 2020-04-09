function SS = spCalculator(yspNominal, dhat, Aaug, Bd, Cd, Ycon, Ucon, Haug, sys)

% Model
A = sys.A;
B = sys.B;
C = sys.C;

% Dimensions
nx = size(A,2);
nu = size(B,2);
ny = size(C,1);

% Objective function weights
Qs = [1, 0; 0, 0.1];

% Define variables
ysp = sdpvar(ny,1);
xsp = sdpvar(nx,1);
usp = sdpvar(nu,1);

% yspNominal = sdpvar(ny,1);
% dhat = sdpvar(nx,1);

% Objective function
objective = (ysp-yspNominal)'*Qs*(ysp-yspNominal);

constraints =  [xsp==A*xsp+B*usp+dhat;
                ysp == C*xsp;
                Ycon.A*ysp<=Ycon.b;
                Ucon.A*usp<=Ucon.b];


% Create optimizer object
ops = sdpsettings('verbose',0);
sol = optimize(constraints,objective,ops);
SS = [value(ysp);value(usp)];

% Calculate the explicit solution using yalmip
% solvemp(constraints,objective ,ops,[yspNominal;dhat],[ysp;usp]);


end

