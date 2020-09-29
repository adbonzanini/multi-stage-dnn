%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script initializes the variables, parameters, and auxiliary
% functions needed to run the plant simulator
%
% Written by: Angelo D. Bonzanini
% Last edited: April 15 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Import casadi
import casadi.*


%% Define the plant simulator

% Sampling time
delta = 0.5;

% Variable size
ndstate = 4;    % additional state for offset-free disturbance
nastate = 5;
ninput = 2;
noutput = 1;    % surface temp
nparam = 1;     % gap distance
nmeas = 2;      % power and temp measured

% Declare variables (use scalar graph)
t  = SX.sym('t');                   % time
u  = SX.sym('u',ninput);            % control
xd  = SX.sym('xd',ndstate);         % differential state
xa  = SX.sym('xa',nastate);         % algebraic state
xddot  = SX.sym('xdot',ndstate);    % differential state time derivative
p = SX.sym('p',nparam);             % parameters

%% Inputs
% Va = u(1)*1e3;
P_set = u(1);
f = 14.0*1e3;
q = u(2);
dsep = p(1)*1.00e-3;

%% Differential states
T1 = xd(1)*300; % Gas plume temperature adjacent to the substrate
w1 = 1.0;  %
Ts_max = xd(2)*300;
intErr = xd(3);

%% Algebraic states
ip = xa(1)*1e-3;     % current in amps  % We convert this to a differential state to learn the GP model
P = xa(2);           % power in W
Rp = xa(3)*1e5;      % plasma resistivity in Ohm
Tin = xa(4)*300;     % peak gas temeprature in K
Va = xa(5)*1e3;      % applied voltage in V

%% Offset-free constant disturbances
v1 = xd(4);

%% Velocity form of PI
Kp = 0.268; %W/kV
tauI = 30*0.001; %seconds
err = P_set-P;
Va_PI = 1e3*(Kp*err+1/tauI*intErr);

%% Constant parameters
% Helium
propHe.cp = 5.1931E+3; %J/kgK
propHe.miu = 2.0484E-5; %kg/m/s
propHe.k = 0.15398; %W/mK
propHe.rho = 0.15608; %kg/m3
propHe.Mw = 4.948e-3;
% Air
propAir.cp = 1.00e3; %J/kgK
propAir.miu = 1.868e-5; %kg/m/s
propAir.k = 0.026; %W/mK
propAir.rho = 1.165; %kg/m3
propAir.Mw = 28.966e-3; 
% Surface
propSurf.rho = 2.8e3;
propSurf.cp = 795.00;
propSurf.k = 1.43;
propSurf.d = 0.20e-3;

% System dimensions
dim.r = 1.5e-3;
dim.vol = 3.1416*1e-2*dim.r^2; %m3 volume of plasma chamber 
dim.Ac = 3.1416*dim.r^2; %m2 flow crossectional area

% Intermediate variables
Tinf = 293.00;  % K ambient temperature
R    = 8.314;   % ideal gas constant
Patm = 101.3e3; % Pa
win  = 1.00;    % inlet He fraction
e1   = 0.90;    % distribution coefficient
eta  = 0.4+0.07*dsep/4.00e-3; % power deposition efficiency
H_I  = 1.6;
rhoin = (Patm/(R))*(propHe.Mw)/Tinf;
vin = q*(1.0/60.0)*0.001*(Tinf/273.0)/dim.Ac;
Pow = (eta*P)/(dim.Ac*1.00e-2);

% Algebraic circuit expressions
n_T = 1.0;
Cp = 0.94003e-11;
b = 20.7911e5;
om = 2.0*3.1416*f;
C0 = 8.072e-12;
eps0 = 8.85e-12; %vacuum permittivity
k_diel = 4.50; %relative premittiviy of quartz
Adiel = 24.00e-3*60e-3; %area of the surface dielectric
Cs = k_diel*eps0*Adiel/2.00e-4; %surface capacitance
Tincalc = Tinf+(dim.r*Pow/(H_I*sqrt(vin)))*((1-e1)+(e1-(1-e1))*exp(-H_I*1.00e-2*sqrt(vin)/(rhoin*vin*propHe.cp*dim.r))-e1*exp(-(2.0*H_I*1.0e-2)*sqrt(vin)/(rhoin*vin*propHe.cp*dim.r)));
Rpcalc = 0.9*(b)*((Tin/340.0));
Pcalc = (Cp^2.00*Cs^2.00*Rp*ip^2.00)/(2.00*(C0^2.00*Cp^2.00*Cs^2.00*Rp^2.00*om^2.00 + C0^2.00*Cp^2.00 + 2.00*C0^2.00*Cp*Cs + C0^2.00*Cs^2.00 + 2.00*C0*Cp^2.00*Cs + 2.00*C0*Cp*Cs^2.00 + Cp^2.00*Cs^2.00));
Vacalc = (ip*((Cp^2.00*Cs^2.00*Rp^2.00*om^2.00 + Cp^2.00 + 2.00*Cp*Cs + Cs^2.00)/(C0^2.00*Cp^2.00*Cs^2.00*Rp^2.00*om^2.00 + C0^2.00*Cp^2.00 + 2.00*C0^2.00*Cp*Cs + C0^2.00*Cs^2.00 + 2.00*C0*Cp^2.00*Cs + 2.00*C0*Cp*Cs^2.00 + Cp^2.00*Cs^2.00))^(1.00/2.00))/om;

% Heat and mass transfer derived experssions
U_h = 1.83*vin^(0.5); %heat transfer in gas
K = 0.017*vin^(0.5); %mass transfer in gas
hgs = 50.0*vin^0.8; %heat transfer between gas and surface
wAinf = 0.0;
n = 1.0;
cp1 = w1*propHe.cp+(1.0-w1)*propAir.cp; %specific heat
rho1 = (Patm/(R))*(propHe.Mw*w1+propAir.Mw*(1.0-w1))/T1; %denisty

% Algebraic equations
f_z = [(Va-Vacalc)/1000.00 ; (P-Pcalc) ; (Rp-Rpcalc)/1.00e5 ; (Tin-Tincalc)/300.00 ; (Va-Va_PI)/1000.0];
f_z_fcn = Function('f_z_fcn',{t,xd,xa,u,p},{f_z});

% Differential equations
dT1dt = (1.00/(rho1*cp1*(dsep/n)*dim.Ac))*(rhoin*vin*cp1*dim.Ac*(Tin-T1)-U_h*(dsep/n)*(2.00*3.1416*dim.r)*(T1-Tinf));
dTs_maxdt = (dim.Ac*hgs*(T1-Ts_max)-2.0*3.1416*propSurf.d*propSurf.k*(Ts_max-Tinf))/(propSurf.rho*propSurf.cp*dim.Ac*propSurf.d);
dintErrdt = err;
dv1dt = 0.0;
f_x = [dT1dt/300.0 ; dTs_maxdt/300.0 ; dintErrdt ; dv1dt];
f_x_fcn = Function('f_x_fcn',{t,xd,xa,u,p},{f_x});

% Output equations
hout = [xd(2)+xd(4)]; 
houtfcn = Function('houtfcn',{t,xd,xa,u,p},{hout});

% Measured variable equations
hmeas = [xd(2)+xd(4) ; xa(2)];  %[Ts_max + disturbance; Power]
hmeasfcn = Function('hmeasfcn',{t,xd,xa,u,p},{hmeas});