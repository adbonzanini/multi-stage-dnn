function myKrigingMat = trainGP(Xtrain, Ytrain, Xtest, Ytest, showFigures)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trains a Gaussian Process (GP) regression model using Xtrain 
% (Nsamp x Nfeatures) and Ytrain (Nsamp x Noutputs). UQLab is requred!
% Set showFigures = 1 if you wish to superimpose the training and test
% predictions with the real data
%
% Written by: Angelo D. Bonzanini
% Last edited: April 17 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start uqlab
% uqlab

% train GP
MetaOpts.Type = 'Metamodel';
MetaOpts.Scaling=0;
MetaOpts.MetaType = 'Kriging';
MetaOpts.Corr.Family = 'matern-3_2';
%MetaOpts.Corr.Family = 'Gaussian';
MetaOpts.Corr.Isotropic = 0;

MetaOpts.ExpDesign.X = Xtrain;
MetaOpts.ExpDesign.Y = Ytrain;
myKrigingMat = uq_createModel(MetaOpts);



% Test Data


if showFigures==1
    [Ypred,Yvar] = uq_evalModel(myKrigingMat,Xtest);
    [YpredTr, YvarTr] = uq_evalModel(myKrigingMat,Xtrain);
    
    
    figure(1)
    % subplot
    subplot(2,2,1)
    plot(Ytrain(:,1), 'r.')
    hold on
    plot(YpredTr(:,1), 'b--')
    ylabel('y_1')
    title('Training Data')
    set(gca,'FontSize',15)
    box on
    % subplot
    subplot(2,2,2)
    plot(Ytrain(:,2), 'r.')
    hold on
    plot(YpredTr(:,2), 'b--')
    ylabel('y_2')
    title('Training Data')
    set(gca,'FontSize',15)
    box on
    % subplot
    subplot(2,2,3)
    plot(Ytest(:,1), 'r.')
    hold on
    plot(Ypred(:,1), 'b--')
    xlabel('Time index')
    ylabel('y_1')
    title('Test Data')
    set(gca,'FontSize',15)
    box on
    % subplot
    subplot(2,2,4)
    plot(Ytest(:,2), 'r.')
    hold on
    plot(Ypred(:,2), 'b--')
    xlabel('Time index')
    ylabel('y_2')
    title('Test Data')
    set(gca,'FontSize',15)
    box on
end

end

