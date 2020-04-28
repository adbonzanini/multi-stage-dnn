function s = casadiDNN(s, H, L)

% extract information
data = s.data_rand';
target = s.target_rand';
Ntr = size(data,2);
nx = size(s.A,1);
np = size(data,1)-nx;

% scale data
data_min = min(data,[],2);
data_max = max(data,[],2);
target_min = min(target,[],2);
target_max = max(target,[],2);
data_s = 2*(data-repmat(data_min,[1,Ntr]))./repmat(data_max-data_min,[1,Ntr])-1;
target_s = 2*(target-repmat(target_min,[1,Ntr]))./repmat(target_max-target_min,[1,Ntr])-1;

% train the network
net = feedforwardnet(H*ones(1,L), 'trainlm');
for l = 1:L
    net.layers{l}.transferFcn = 'poslin';
end
net.trainParam.showWindow = 0; 
[net,tr] = train(net, data_s, target_s);

% extract weights and biases from neural network
W = cell(L+1,1);
b = cell(L+1,1);
W{1} = net.IW{1};
b{1} = net.b{1};
for i = 1:L
    W{i+1} = net.LW{i+1,i};
    b{i+1} = net.b{i+1};
end

% create casadi evaluation of neural network
import casadi.*
x = MX.sym('x', nx+np);
xs = 2*(x-data_min)./(data_max-data_min)-1;
z = max(W{1}*xs+b{1},0);
for k = 1:L-1
    z = max(W{k+1}*z+b{k+1},0);
end
us = W{L+1}*z+b{L+1};
u = (us+1)/2.*(target_max-target_min)+target_min;
%u = min(max(u,s.u_min'),s.u_max');
dnnmpc = Function('dnnmpc', {x}, {u});

% store information
s.data_min = data_min;
s.data_max = data_max;
s.target_min = target_min;
s.target_max = target_max;
s.net = net;
s.dnnmpc = dnnmpc;

end