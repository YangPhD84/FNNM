%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is a demo of the FNNM method on drug-target interaction  prediction.
% Nuclear Receptors, GPCRs, Ion Channels, and Enzymes are downloaded from [1].
%
% [1] Y. Yamanishi, M. Araki, A. Gutteridge, W. Honda, and M. Kanehisa.
% Prediction of drug-target interaction networks from the integration of
% chemical and genomic spaces. Bioinformatics, 24(13):i232¨Ci240, 2008.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1: Load Data (Nuclear Receptors, GPCRs, Ion Channels, and Enzymes)
clear
R = 5;    % Number of independent simulations
K = 10;   % K-fold cross validation
data = 4  % Dataset to run, change this value for running different datasets
if data == 1
    dg = dlmread('drug_target_datasets/en_sim_dg.txt');
    dc = dlmread('drug_target_datasets/en_sim_dc.txt');
    adj = dlmread('drug_target_datasets/en_adj.txt');
elseif data == 2
    dg = dlmread('drug_target_datasets/ic_sim_dg.txt');
    dc = dlmread('drug_target_datasets/ic_sim_dc.txt');
    adj = dlmread('drug_target_datasets/ic_adj.txt');
elseif data == 3
    dg = dlmread('drug_target_datasets/gpcr_sim_dg.txt');
    dc = dlmread('drug_target_datasets/gpcr_sim_dc.txt');
    adj = dlmread('drug_target_datasets/gpcr_adj.txt');
elseif data == 4
    dg = dlmread('drug_target_datasets/nr_sim_dg.txt');
    dc = dlmread('drug_target_datasets/nr_sim_dc.txt');
    adj = dlmread('drug_target_datasets/nr_adj.txt');
end

addpath('FNNM');
addpath('functions');
XX = fPCA(dc, 0.9);
YY = fPCA(dg, 0.9);
tol1 = 1e-4;
maxiter = 3000;
mu1 = 0.1;
mu2 = mu1;
%% STEP 2:  Choose Parameters
rng('default')
k0 = 1;
for lamda1 = [0.001 0.01 0.1 1 10 100 1000];
    for lamda2 = [0.001 0.01 0.1 1 10 100 1000];
        precision_SD = []; recall_SD = [];
        aucSD = zeros(1,K);
        auprSD = zeros(1,K);
        auc_SD = zeros(1,R);
        aupr_SD = zeros(1,R);
        
        y = adj;
        crossval_idx = crossvalind('Kfold', y(:), K);
        
        for i = 1 : K   % each fold at a time
            train_idx = find(crossval_idx ~= i);
            test_idx  = find(crossval_idx == i);
            
            y_train = y;
            y_train(test_idx) = 0;
            train = y_train;
            test = y - train;
            
            yy = y;
            yy(yy == 0) = -1;
            
            [Z_D, A, B, iter] = FNNM(train, YY, XX, lamda1, lamda2, mu1, mu2, maxiter, tol1);
            statsSD = evaluate_performance(Z_D(test_idx), yy(test_idx), 'classification');
            aucSD(i) = statsSD.auc;
            auprSD(i) = statsSD.aupr;
        end
        auc_SD = mean(aucSD);
        aupr_SD = mean(auprSD);
        
        result(k0, :) = [auc_SD+aupr_SD, auc_SD, aupr_SD, lamda1, lamda2, mu1, mu2, iter, tol1, maxiter];
        k0 = k0 + 1;
    end
end

bestparas_num = find(result(:, 1) == max(result(:, 1)))
lamda1 = result(bestparas_num, 4)
lamda2 = result(bestparas_num, 5)
%% STEP 3:  10-fold Cross Validation for 5 Times
rng('default')
precision_SD = []; recall_SD = [];
aucSD = zeros(1,K);
auprSD = zeros(1,K);
auc_SD = zeros(1,R);
aupr_SD = zeros(1,R);

for r = 1 : R        % Number of simulations
    y = adj;
    crossval_idx = crossvalind('Kfold', y(:), K);
    for i = 1 : K    % each fold at a time
        train_idx = find(crossval_idx ~= i);
        test_idx  = find(crossval_idx == i);
        
        y_train = y;
        y_train(test_idx) = 0;
        train = y_train;
        test = y - train;
        
        yy = y;
        yy(yy == 0) = -1;
        
        [Z_D, A, B, iter] = FNNM(train, YY, XX, lamda1, lamda2, mu1, mu2, maxiter, tol1);
        statsSD = evaluate_performance(Z_D(test_idx), yy(test_idx), 'classification');
        aucSD(i) = statsSD.auc;
        auprSD(i) = statsSD.aupr;
    end
    auc_SD(r) = mean(aucSD);
    aupr_SD(r) = mean(auprSD);
end

FinalAUC = mean(auc_SD);
FinalAUPR = mean(aupr_SD);

Paras.lamda1 = lamda1;
Paras.lamda2 = lamda2;
Paras.mu1 = mu1;
Paras.mu2 = mu2;
Paras.iter = iter;
Paras.tol1 = tol1;
Paras.maxiter = maxiter;
%% STEP 4: Print AUC and AUPR Results
disp(['The final AUC: ' num2str(FinalAUC)]);
disp(['The final AUPR: ' num2str(FinalAUPR)]);
fprintf('\n');