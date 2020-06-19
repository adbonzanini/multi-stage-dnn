function Kmat = kernelFn(XN,XM,theta)

Kmat = (exp(theta(2))^2)*exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2))+(1e-4)*eye(size(XN,1), size(XM,1));


end
