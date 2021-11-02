% set root dir as cw1

close all
clear
clc

addpath('./datasets');
addpath('./functions');

rng(1);

%% load data

heart_table = readtable('./datasets/heart_failure_clinical_records_dataset.csv');
head(heart_table) % preview top 8 rows of data

%% split features and labels

heart_mat = table2array(heart_table);
heart_X = heart_mat(:, 1:end-1);
heart_Y = heart_mat(:, end);

%% feature normalisation
heart_X = minMaxNorm(heart_X);

%% SVM linear kernal
partitionedModel = fitcsvm(heart_X, heart_Y, 'KernelFunction','linear', 'BoxConstraint',1, "KFold", 10);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
disp("Accuracy: " + validationAccuracy);
