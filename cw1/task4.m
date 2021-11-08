%% Q1

% clean workspace
close all
clear
clc

addpath('./datasets');
addpath('./functions');

rng(1);

% data loading and normalisation
heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
heart_mat = table2array(heart_table);
heart_mat = shuffleRows(heart_mat);

heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

heart_X = minMaxNorm(heart_X);
norm_heart_mat = [heart_X heart_Y];


instances = height(norm_heart_mat);
train_portion = round(instances * 0.8);


train_mat = norm_heart_mat(1:train_portion, :);
test_mat = norm_heart_mat(train_portion + 1 : end,:);

figure,
hist(norm_heart_mat(:, end));

% training
sigmas = [0.01:0.01:10];
iter = length(sigmas);
f1Score = zeros(1, iter);
accuracies = zeros(1, iter);

for i = 1:length(sigmas)

    model = fitcsvm( ...
    train_mat(:, 1:end-1), ...
    train_mat(:, end), ...
    "KernelFunction", "gaussian", ...
    "BoxConstraint", 10, ...
    "KernelScale", sigmas(i), ...
    "Verbose", 1 ...
    );
    f1Score(1, i) = myFOneScore(model, test_mat(:, 1:end-1), test_mat(:, end));
    accuracies(1, i) = myAccuracy(model, test_mat(:, 1:end-1), test_mat(:, end));
    
end

figure,
plot(sigmas, f1Score);
figure,
plot(sigmas, accuracies);
