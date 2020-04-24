function Fsim = plantSimulator(xd_in, u_in, d_in, xa_in)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% High-fidelity model of an atmospheric pressure plasma jet in Helium.
% Required CasADi for MATLAB to be installed and imported.
%
% FUNCTION INPUTS:
% xd_in (differential states at time k) = [T1/300; Ts_max/300; intErr; v1]        
% u_in (system inputs at time k) = [P_set; q] 
% d_in (disturbance at time k) = [dsep]
% xa_in (algebraic states at time k) = [ip*1e3; P; Rp*1e-5; Tin/300; Va*1e-3]
%
% FUNCTION OUTPUTS:
% Fsim: struct with predictions at time k+1
% Fsim.xf: differential states at time k+1
% Fsim.zf: algebraic states at time k+1
%
% Written by: Angelo D. Bonzanini and Joel A. Paulson
% Last edited: April 15 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import casadi.*
% Run initialization to define variables, parameters, etc
initializePlant;

% Integrator
dae = struct('x', xd, 'z', xa, 'p', vertcat(u,p), 'ode', f_x, 'alg', f_z);
opts_dae = struct('tf', delta);
F = integrator('F', 'idas', dae, opts_dae);


Fsim = F('x0',xd_in,'p',[u_in; d_in],'z0',xa_in);
end


