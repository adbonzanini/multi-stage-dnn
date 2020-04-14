%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Summary: Converts ARMAX model to non-minimal state-space to use in MPC
% This script follows the notation of Wang and Yound, 2005 (IFAC World
% Congress)

% Written by:   Angelo D. Bonzanini
% Date:         April 10 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear workspace
clear all

% Define number of inputs (p) and outputs (q)
p = 2;
q = 2;

%% Load ARMAX model
model = load('APPJarmaxGlass');
AX = model.AX;
BX = model.BX;
CX = model.CX;

% Determine the model order
order = size(AX{1},2)-1;

%% Create matrices A1, A2, ... An, and B1, B2, ..., Bn

% Define empty matrices 
A = cell(order,1);
B = cell(order,1);
C = cell(order,1);

% Ignore the first entry of the matrices to obtain the same form of the model in the paper
for j = 1:order
    A{j} = [AX{1,1}(1,j+1), AX{1,2}(1,1);
            AX{2,1}(1,1), AX{2,2}(1,j+1)];
    
    B{j} = [BX{1,1}(1,j+1), BX{1,2}(1,j+1);
            BX{2,1}(1,j+1), BX{2,2}(1,j+1)]; 
end
    
%% Stack matrices to create Am, Bm, Cm for the
% state vector Dx = [Dy_k; ...; Dy_{k-n+1}; Du_k;... Du_{k-n+1}]

% [A1, A2, ..., An]
A1_to_n = [A{1:end}];
% [B1, B2, ..., Bn]
B1_to_n = [B{2:end}];

% Identity matrices
topLeft = eye(q*order, q*order);
topLeft(end-q+1:end,:) = zeros(size(topLeft(end-q+1:end,:)));

topRight = zeros(q*order, q*(order-1));

bottomRight = eye(p*(order-1)-p, p*(order-1));
bottomLeft = zeros(p*(order-1)-p, q*order);

% Stack matrices
Am = [-A1_to_n, B1_to_n;
     topLeft, topRight; 
     bottomLeft, bottomRight];

Bm = [B{1};
      zeros(p*(order-1), p);
      eye(p);
      zeros(p*(order-1)-p)];
  
Cm = [eye(p), zeros(q, size(Am,1)-p)];

A = Am;
B = Bm;
C = Cm;
%{
%% Choose new state variable vector x = [Dx; y] and define A, B, and C
A = [Am, zeros(size(Am, 1), q);
     Cm*Am, eye(q)];

B = [Bm; Cm*Bm];

C = [zeros(q, size(Am,1)), eye(q)];
%}

%% Choose new state variable vector x = [Dx; y] and define A, B, and C
Nx = q*order+p*(order-1);
Ae = [A, zeros(Nx, q);-A(1:q,1:end), eye(q)];
Be=[B;-B(1:q,:);];
Ce=[C,zeros(q, q)];
Ew = [zeros(Nx, q);-B(1:q,:)];


% Update the model
model.A = Ae;
model.B = Be;
model.C = Ce;
model.Ew = Ew;


save('APPJ_NMSS', 'model')


