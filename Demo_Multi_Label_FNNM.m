%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is a demo of the FNNM method on multi-label learning.
% Arts is downloaded from [1].
%
% [1] N. Ueda, K. Saito. Parametric mixture models for multi-label text.
% In: NIPS, 2003, 737-744.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1: Loading Data (Arts)
clear
rng('default')
addpath('FNNM');
addpath('functions');
dataname='Arts';
data=['multi_label_dataset/' dataname '.mat'];
load(data);
times = 5;
percent = 0.1;

U = [train_data; test_data];
hatT = [train_target; test_target];
[n, q] = size(hatT);

%% STEP 2: Generating Omega
train_num = round(n * 0.9);
% Generate 90% training and 10% testing
obrT = zeros(size(hatT));
indexperm = randperm(n);
train_index = indexperm(1, 1:train_num);
test_index = indexperm(1, train_num+1:n);
remainT = hatT(train_index, :);
% Generate the percent% randomly observed entries in the training data
for iii = 1 : q
    positive_index = find(remainT(:, iii) > 0);
    positive_number = length(positive_index);
    positive_random = randperm(positive_number);
    positive_select = positive_index(positive_random(1, 1:ceil(positive_number * percent)), 1);
    negative_index = find(remainT(:, iii) <= 0);
    negative_number = length(negative_index);
    negative_random = randperm(negative_number);
    negative_select = negative_index(negative_random(1, 1:ceil(negative_number * percent)), 1);
    obrT(train_index(1, positive_select), iii) = 1;
    obrT(train_index(1,negative_select),iii)=1;
end

if  min(min(hatT)) == -1
    hatT = (hatT + 1) / 2;
end

[A, R_A] = qr(U, 0);
M_Omega = hatT .* obrT;
B = eye(size(M_Omega, 2));

%% STEP 3: Choosing Parameters
tol1 = 1e-4;
maxiter = 3000;
mu1 = 0.1;
mu2 = mu1;
seq_lambda1 =[ 1e-3 1e-2 1e-1 1 10 100 1000 ];
seq_lambda2 =[ 1e-3 1e-2 1e-1 1 10 100 1000 ];
need = zeros( length( seq_lambda1), length( seq_lambda2));

for i = 1 : length( seq_lambda1 )
    for j = 1: length( seq_lambda2)
        [T] = FNNM(M_Omega, A, B, seq_lambda1(i), seq_lambda2(j), mu1, mu2, maxiter, tol1);
        results = AveragePrecision(T, hatT)
        need(i, j) = results(1, 1)
    end
end

index = find(need == max(max(need)));
[I_row, I_col] = ind2sub(size(need), index);
lambda1 = seq_lambda1( I_row(1) )
lambda2 = seq_lambda2( I_col(1) )

%% STEP 4: Running for 5 times
rng('default')
ko = 1;
for percent = 0.1 : 0.2 : 0.9
    for k = 1 : times
        train_num = round(n * 0.9);
        %Generate 90% training and 10% testing
        obrT = zeros(size(hatT));
        indexperm = randperm(n);
        train_index = indexperm(1, 1:train_num);
        test_index = indexperm(1, train_num+1:n);
        remainT = hatT(train_index, :);
        %Generate the percent% randomly observed entries in the training data
        for iii = 1:q
            positive_index = find(remainT(:, iii) > 0);
            positive_number = length(positive_index);
            positive_random = randperm(positive_number);
            positive_select = positive_index(positive_random(1, 1:ceil(positive_number * percent)), 1);
            negative_index = find(remainT(:, iii) <= 0);
            negative_number = length(negative_index);
            negative_random = randperm(negative_number);
            negative_select = negative_index(negative_random(1, 1:ceil(negative_number * percent)), 1);
            obrT(train_index(1, positive_select), iii) = 1;
            obrT(train_index(1, negative_select), iii) = 1;
        end
        
        if  min(min(hatT)) == -1
            hatT = (hatT + 1) / 2;
        end
        
        [A, R_A] = qr(U, 0);
        M_Omega = hatT .* obrT;
        B = eye(size(M_Omega, 2));
        [T] = FNNM(M_Omega, A, B, lambda1, lambda2, mu1, mu2, maxiter, tol1);
        need1(k) = AveragePrecision(T, hatT);%AP on all data
    end
    
    Results(ko, :)=[percent, mean(need1), std(need1), lambda1, lambda2, mu1, mu2, maxiter, tol1];
    ko = ko + 1;
end

%% STEP 5: Saving Results
save Arts_FNNM Results
