function [gprMdl1, gprMdl2, kfcn] = trainGP(Xtrain, Ytrain, Xtest, Ytest, showFigures)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trains a Gaussian Process (GP) regression model using Xtrain 
% (Nsamp x Nfeatures) and Ytrain (Nsamp x Noutputs). UQLab is requred!
% Set showFigures = 1 if you wish to superimpose the training and test
% predictions with the real data
%
% Written by: Angelo D. Bonzanini
% Last edited: April 17 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define anonymous function of the kernel
kfcn = @(XN,XM,theta) (exp(theta(2))^2)*exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2))+(1e-2)*eye(size(XN,1), size(XM,1));

theta0 = [1.5,0.2];

% Train each dimension
gprMdl1 = fitrgp(Xtrain, Ytrain(:,1),'KernelFunction',kfcn, 'KernelParameters',theta0);
gprMdl2 = fitrgp(Xtrain, Ytrain(:,2),'KernelFunction',kfcn, 'KernelParameters',theta0);

%%

if showFigures==1
    [ypred1, ypred1SD] = predict(gprMdl1, Xtest);
    [ypred2, ypred2SD] = predict(gprMdl2, Xtest);

    theta1 = gprMdl1.KernelInformation.KernelParameters;  %(sigmaL, sigmaF)
    KK = kfcn(Xtrain, Xtrain, theta1);
    KKs = kfcn(Xtrain, Xtest, theta1);
    ypred1m = (KKs'/KK)*Ytrain(:,1);

    theta2 = gprMdl2.KernelInformation.KernelParameters;  %(sigmaL, sigmaF)
    KK = kfcn(Xtrain, Xtrain, theta2);
    KKs = kfcn(Xtrain, Xtest, theta2);
    ypred2m = (KKs'/KK)*Ytrain(:,2);

    figure()
    subplot(2,1,1)
    hold on
    plot(ypred1)
    plot(Ytest(:,1), 'k.')
    plot(ypred1m)
    subplot(2,1,2)
    ylabel('y_2')
    hold on
    plot(ypred2)
    plot(Ytest(:,2), 'k.')
    plot(ypred2m)
    ylabel('y_2')
    xlabel('Discrete Time')
end




end

